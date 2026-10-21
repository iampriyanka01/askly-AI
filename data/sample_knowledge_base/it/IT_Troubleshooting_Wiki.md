---
title: IT Troubleshooting Wiki
category: it
url: https://wiki.company.com/it/troubleshooting
owner: IT Service Desk
last_updated: 2026-05-12
---

# IT Troubleshooting Wiki

Self-help guides for the most common IT issues. If a guide does not solve your problem, raise a ticket in the IT Service Portal (servicedesk.company.com) or call the IT Service Desk on extension 4357. The Service Desk is staffed 07:00 to 21:00 on weekdays; priority 1 incidents are handled 24/7.

## 1 VPN (GlobalProtect)

### 1.1 Connecting to the VPN
Acme Corp uses GlobalProtect for remote access to internal systems. Open the GlobalProtect app from the system tray (Windows) or menu bar (macOS), enter the portal address vpn.company.com and sign in with your SSO credentials and authenticator approval. The VPN is required for the HR Portal, finance systems and internal wikis when you are outside the office.

### 1.2 VPN Will Not Connect
If the VPN will not connect: (1) check that you have a working internet connection by opening any public website; (2) quit and reopen GlobalProtect; (3) make sure the portal address is exactly vpn.company.com; (4) restart your laptop; (5) if you are on hotel or airport Wi-Fi, complete the captive portal sign-in page first. If the error message is "Gateway not responding", try selecting a different gateway region in the app settings. If none of this works, raise a ticket with a screenshot of the error.

### 1.3 VPN Is Slow
Slow VPN connections are usually caused by a distant gateway. Choose the "Best Available" gateway, close large downloads or video streaming, and use a wired connection where possible. Video calls in Teams and Zoom do not need the VPN and run faster with it disconnected.

## 2 Wi-Fi in the Office

### 2.1 Connecting to Office Wi-Fi
Company laptops connect automatically to the "Acme-Corp" network using a device certificate. Personal phones and guests should use "Acme-Guest", whose daily password is shown on the screens at reception.

### 2.2 Wi-Fi Keeps Dropping
Forget the "Acme-Corp" network, restart the laptop and reconnect. If it still drops, check for pending operating system updates, as outdated wireless drivers are the most common cause. Report building-wide outages to facilities@company.com.

## 3 Email and Outlook

### 3.1 Outlook Not Syncing
If Outlook is not receiving new email: check the status bar for "Working Offline" and turn it off under Send/Receive; restart Outlook; and check the IT status page at status.company.com for known outages. If your mailbox is full (over 100 GB), archive old items using File > Tools > Mailbox Cleanup.

### 3.2 Shared Mailboxes
Access to a shared mailbox is requested through the IT Service Portal under Request > Email > Shared Mailbox Access and must be approved by the mailbox owner. Once approved, the mailbox appears in Outlook automatically within 2 hours.

### 3.3 Out-of-Office Replies
Set an automatic reply in Outlook under File > Automatic Replies, or in Outlook on the web under Settings > Mail > Automatic replies.

## 4 Laptops and Devices

### 4.1 Laptop Running Slowly
Restart your laptop at least once a week so updates can install. Check Task Manager (Windows) or Activity Monitor (macOS) for applications using a lot of memory. Make sure at least 15% of your disk is free. Laptops older than 4 years are eligible for a hardware refresh, which can be requested in the IT Service Portal.

### 4.2 Lost or Stolen Laptop
Report a lost or stolen laptop to the IT Service Desk immediately, even outside office hours. IT will remotely lock and wipe the device. You will also need to file a police report if the device was stolen, and inform your manager.

### 4.3 Requesting New Hardware
Monitors, keyboards, headsets and other accessories are requested in the IT Service Portal under Request > Hardware. Standard accessories are delivered within 5 working days; non-standard items need manager approval.

## 5 Software

### 5.1 Installing Software
Approved software can be installed without admin rights from the Company Portal app (Windows) or Self Service app (macOS). Software that is not in the catalogue must be requested in the IT Service Portal under Request > Software; requests are reviewed by IT Security for licensing and security within 3 working days.

### 5.2 Microsoft Teams Problems
If Teams is not loading, sign out, clear the cache (Settings > General > Clear cache, or delete the Teams cache folder) and sign back in. For audio problems, select the correct microphone and speaker under Settings > Devices and run a test call.

## 6 Printing

### 6.1 Adding a Printer
Office printers use follow-me printing. Install the "Acme-Print" queue from the Company Portal or Self Service app. Send your job to Acme-Print and release it at any printer by tapping your badge.

### 6.2 Printer Not Working
If your job does not appear at the printer, check that you are connected to the Acme-Corp Wi-Fi or VPN, and that you tapped the badge you are registered with. Paper jams and toner problems should be reported to facilities@company.com.
