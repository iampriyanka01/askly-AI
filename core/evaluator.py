"""
core/evaluator.py  -  QUALITY MEASUREMENT with RAGAS (+ free offline proxies)
=============================================================================

Askly doesn't just answer - it measures its own quality, so changes to chunking,
retrieval or prompts can be compared with numbers instead of impressions.

    data/eval/testset.json  (question, optional history, ground_truth, expected_sources)
        -> AsklyAssistant.ask()         real answers + the real retrieved contexts
        -> RAGAS (LLM judge)            5 metrics, per question and averaged
        -> retrieval metrics (no LLM)   source hit-rate, MRR, gap-detection accuracy
        -> data/state/evaluations.csv   history for comparing runs

RAGAS metrics
    faithfulness        is every claim in the answer supported by the retrieved context?
    answer_relevancy    does the answer address the question?
    context_precision   are the relevant chunks ranked at the top?
    context_recall      does the retrieved context contain what the reference answer needs?
    answer_correctness  factual + semantic agreement with the reference answer

Offline proxies (no API key needed - clearly labelled, NOT RAGAS)
    token-overlap approximations of the same five ideas, so the evaluation page and
    CLI still produce a comparable signal while developing without a key.

judge_failed: True when RAGAS could not score a metric at all (quota, timeout,
unparseable judge reply) - a 0.0 then means "unknown", not "bad" (HireFlow HW21).

Public API
    RAGEvaluator(assistant).estimate_cost(n) / run(testset_path, limit, use_ragas)
    RAGEvaluator.score_answer(question, answer, contexts)  - faithfulness + relevancy for ONE turn
    get_evaluation_summary() / export_evaluations()
"""

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from utils.config import PROJECT_ROOT, STATE_DIR, LLM_REQUESTS_PER_MINUTE
from utils.utils import get_logger, tokenize, split_sentences, get_llm, get_embeddings

logger = get_logger(__name__)

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]
WEIGHTS = {"faithfulness": 0.30, "answer_relevancy": 0.20, "context_precision": 0.15,
           "context_recall": 0.15, "answer_correctness": 0.20}
DEFAULT_TESTSET = PROJECT_ROOT / "data" / "eval" / "testset.json"


@dataclass
class EvaluationReport:
    method: str                                  # "ragas" | "offline_proxy"
    metrics: Dict[str, float]
    overall_score: float
    retrieval: Dict[str, float]
    rows: List[Dict[str, Any]] = field(default_factory=list)
    judge_failed: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _f1(pred: str, ref: str) -> float:
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r:
        return 0.0
    common = sum(min(p.count(t), r.count(t)) for t in set(p))
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


class RAGEvaluator:
    """Evaluates the full Askly pipeline on a labelled test set."""

    def __init__(self, assistant):
        self.assistant = assistant
        self.history: List[EvaluationReport] = []
        self.csv_path = Path(STATE_DIR) / "evaluations.csv"

    # ------------------------------------------------------------------ cost

    @staticmethod
    def estimate_cost(n_questions: int, use_ragas: bool = True) -> Dict[str, Any]:
        """Roughly: Askly itself uses 1-3 LLM calls per question (analysis, rerank,
        answer), RAGAS ~2 per metric per question. Shown BEFORE running."""
        askly_calls = 3 * n_questions
        ragas_calls = (len(METRICS) * 2 * n_questions) if use_ragas else 0
        total = askly_calls + ragas_calls
        return {"estimated_llm_calls": total,
                "estimated_minutes": round(total / max(LLM_REQUESTS_PER_MINUTE, 1), 1)}

    # ------------------------------------------------------------------ data

    @staticmethod
    def load_testset(path: Optional[str] = None) -> List[Dict[str, Any]]:
        return json.loads(Path(path or DEFAULT_TESTSET).read_text(encoding="utf-8"))

    def _collect(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run every test question through the real assistant, in a fresh session."""
        rows = []
        for item in items:
            sid = self.assistant.memory.new_session({"user_type": item.get("user_type", "employee")})
            for prev in item.get("history", []):          # replay earlier turns for follow-up items
                self.assistant.ask(prev, sid)
            resp = self.assistant.ask(item["question"], sid)
            cited_sources = [c.source for c in resp.citations]
            expected = item.get("expected_sources", [])
            rank = next((i for i, s in enumerate(cited_sources, start=1) if s in expected), None)
            rows.append({
                "id": item.get("id"), "question": item["question"], "standalone_query": resp.standalone_query,
                "answer": resp.answer, "contexts": resp.contexts or ["(no context retrieved)"],
                "ground_truth": item["ground_truth"], "expected_sources": expected,
                "cited_sources": cited_sources, "source_hit": bool(rank) if expected else None,
                "reciprocal_rank": (1 / rank if rank else 0.0) if expected else None,
                "expect_gap": bool(item.get("expect_gap")), "knowledge_gap": resp.knowledge_gap,
                "confidence": resp.confidence, "generation_mode": resp.generation_mode,
            })
        return rows

    # ------------------------------------------------------------------ run

    def run(self, testset_path: Optional[str] = None, limit: Optional[int] = None,
            use_ragas: bool = True) -> EvaluationReport:
        items = self.load_testset(testset_path)[: limit or None]
        use_ragas = use_ragas and get_llm() is not None
        est = self.estimate_cost(len(items), use_ragas)
        if use_ragas:
            logger.warning(f"Evaluating {len(items)} questions: ~{est['estimated_llm_calls']} LLM calls, "
                           f"~{est['estimated_minutes']} min at {LLM_REQUESTS_PER_MINUTE} requests/minute.")
        rows = self._collect(items)
        retrieval = self._retrieval_metrics(rows)
        # Gap questions have no "right" context, so they're scored only by gap accuracy.
        answerable = [r for r in rows if not r["expect_gap"]]

        report = None
        if use_ragas and answerable:
            report = self._ragas(answerable, retrieval)
        if report is None:
            report = self._proxy(answerable, retrieval)
        for r in rows:
            if r["expect_gap"]:
                r["gap_correct"] = r["knowledge_gap"]
                report.rows.append(r)
        self.history.append(report)
        self.export_evaluations(report)
        return report

    def _retrieval_metrics(self, rows: List[Dict[str, Any]]) -> Dict[str, float]:
        with_src = [r for r in rows if r["expected_sources"]]
        gap_rows = [r for r in rows if r["expect_gap"]]
        false_gaps = [r for r in rows if not r["expect_gap"] and r["knowledge_gap"]]
        return {
            "source_hit_rate": round(sum(r["source_hit"] for r in with_src) / len(with_src), 3) if with_src else 0.0,
            "mrr": round(sum(r["reciprocal_rank"] for r in with_src) / len(with_src), 3) if with_src else 0.0,
            "gap_detection_accuracy": round(sum(r["knowledge_gap"] for r in gap_rows) / len(gap_rows), 3) if gap_rows else 1.0,
            "false_gap_rate": round(len(false_gaps) / max(len(rows) - len(gap_rows), 1), 3),
        }

    def _ragas(self, rows: List[Dict[str, Any]], retrieval: Dict[str, float]) -> Optional[EvaluationReport]:
        try:
            from ragas import evaluate
            from ragas.metrics import (faithfulness, answer_relevancy, context_precision,
                                       context_recall, answer_correctness)
            dataset = self._to_dataset(rows)
            result = evaluate(dataset=dataset,
                              metrics=[faithfulness, answer_relevancy, context_precision, context_recall,
                                       answer_correctness],
                              llm=self._judge_llm(), embeddings=self._judge_embeddings(),
                              run_config=self._run_config(), show_progress=False, raise_exceptions=False)
            table = result.to_pandas()
        except Exception as e:
            logger.error(f"RAGAS evaluation failed ({e}); falling back to offline proxies")
            return None
        metrics, failed = {}, False
        for m in METRICS:
            mean, f = self._column_mean(table, m)
            metrics[m], failed = round(mean, 3), failed or f
        for row, (_, t) in zip(rows, table.iterrows()):
            for m in METRICS:
                v = t.get(m)
                row[m] = None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 3)
        return EvaluationReport("ragas", metrics, self._overall(metrics), retrieval, rows, judge_failed=failed)

    def _proxy(self, rows: List[Dict[str, Any]], retrieval: Dict[str, float]) -> EvaluationReport:
        """Token-overlap stand-ins for the RAGAS metrics (offline, deterministic)."""
        for r in rows:
            ctx = " ".join(r["contexts"])
            ctx_tokens = set(tokenize(ctx))
            sents = [s for s in split_sentences(r["answer"].replace("\n", " ")) if tokenize(s)]
            supported = [s for s in sents if len(set(tokenize(s)) & ctx_tokens) / len(set(tokenize(s))) >= 0.7]
            q = set(tokenize(r["standalone_query"]))
            ref = set(tokenize(r["ground_truth"]))
            rel_flags = [len(set(tokenize(c)) & ref) / max(len(ref), 1) >= 0.3 for c in r["contexts"]]
            hits, ap = 0, 0.0
            for k, flag in enumerate(rel_flags, start=1):
                if flag:
                    hits += 1
                    ap += hits / k
            r.update({
                "faithfulness": round(len(supported) / len(sents), 3) if sents else 0.0,
                "answer_relevancy": round(len(q & set(tokenize(r["answer"]))) / max(len(q), 1), 3),
                "context_precision": round(ap / hits, 3) if hits else 0.0,
                "context_recall": round(len(ref & ctx_tokens) / max(len(ref), 1), 3),
                "answer_correctness": round(_f1(r["answer"], r["ground_truth"]), 3),
            })
        metrics = {m: round(sum(r[m] for r in rows) / len(rows), 3) if rows else 0.0 for m in METRICS}
        return EvaluationReport("offline_proxy", metrics, self._overall(metrics), retrieval, rows)

    # ------------------------------------------------------------------ single answer

    def score_answer(self, question: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
        """Faithfulness + answer relevancy for ONE live answer (~4-6 LLM calls)."""
        if get_llm() is None:
            row = {"standalone_query": question, "answer": answer, "contexts": contexts or [""],
                   "ground_truth": answer}
            self._proxy([row], {})
            return {"method": "offline_proxy", "faithfulness": row["faithfulness"],
                    "answer_relevancy": row["answer_relevancy"]}
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy
            from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
            ds = EvaluationDataset(samples=[SingleTurnSample(user_input=question, response=answer,
                                                             retrieved_contexts=contexts or [""])])
            table = evaluate(dataset=ds, metrics=[faithfulness, answer_relevancy], llm=self._judge_llm(),
                             embeddings=self._judge_embeddings(), run_config=self._run_config(),
                             show_progress=False).to_pandas()
            out = {"method": "ragas"}
            for m in ("faithfulness", "answer_relevancy"):
                mean, failed = self._column_mean(table, m)
                out[m] = None if failed else round(mean, 3)
            return out
        except Exception as e:
            logger.error(f"Single-answer RAGAS scoring failed: {e}")
            return {"method": "ragas", "error": str(e)}

    # ------------------------------------------------------------------ RAGAS plumbing

    @staticmethod
    def _to_dataset(rows: List[Dict[str, Any]]):
        from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
        return EvaluationDataset(samples=[SingleTurnSample(
            user_input=r["standalone_query"], retrieved_contexts=list(r["contexts"]),
            response=r["answer"], reference=r["ground_truth"]) for r in rows])

    @staticmethod
    def _judge_llm():
        from ragas.llms import LangchainLLMWrapper
        return LangchainLLMWrapper(get_llm(temperature=0.0))   # shares the app's rate limiter

    @staticmethod
    def _judge_embeddings():
        from ragas.embeddings import LangchainEmbeddingsWrapper
        emb = get_embeddings()
        return LangchainEmbeddingsWrapper(emb) if emb else None

    @staticmethod
    def _run_config():
        """RAGAS defaults to 16 parallel workers; on a 5 requests/minute free tier that
        means 429s, then timeouts, then NaN everywhere. One worker, patient retries."""
        from ragas.run_config import RunConfig
        return RunConfig(max_workers=1, timeout=900, max_retries=10, max_wait=60)

    @staticmethod
    def _column_mean(table: pd.DataFrame, column: str):
        if column not in table.columns:
            return 0.0, True
        values = table[column].dropna()
        return (float(values.mean()), False) if len(values) else (0.0, True)

    @staticmethod
    def _overall(metrics: Dict[str, float]) -> float:
        return round(sum(metrics.get(m, 0.0) * w for m, w in WEIGHTS.items()), 3)

    # ------------------------------------------------------------------ history

    def get_evaluation_summary(self) -> Dict[str, Any]:
        if not self.history:
            return {"runs": 0}
        last = self.history[-1]
        return {"runs": len(self.history), "last_method": last.method, "last_overall": last.overall_score,
                "last_metrics": last.metrics, "last_retrieval": last.retrieval}

    def export_evaluations(self, report: EvaluationReport) -> bool:
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            row = {"timestamp": report.timestamp, "method": report.method, "overall": report.overall_score,
                   "judge_failed": report.judge_failed, **report.metrics, **report.retrieval,
                   "questions": len(report.rows)}
            df = pd.DataFrame([row])
            df.to_csv(self.csv_path, mode="a", header=not self.csv_path.exists(), index=False)
            return True
        except Exception as e:
            logger.error(f"Could not export evaluation: {e}")
            return False

    def load_past_runs(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.csv_path)
        except Exception:
            return pd.DataFrame()
