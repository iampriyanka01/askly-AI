"""Test setup: isolate state files and force offline mode (no API calls in tests)."""

import os
import sys
import tempfile
from pathlib import Path

_STATE = tempfile.mkdtemp(prefix="askly-test-state-")
os.environ["STATE_DIR"] = _STATE
os.environ["GOOGLE_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["FAISS_INDEX_DIR"] = str(Path(_STATE) / "faiss")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
