---
name: Don't suppress long-running output
description: User wants to see live output from trace collection runs, not run them in background
type: feedback
---

Don't run trace collection scripts (start_*.sh, collect_traces.py) in the background with run_in_background. The user wants to see the live output scrolling by. Just run them normally with a long timeout.

**Why:** When run in background, the user can't see what the model is doing — they lose visibility into progress and errors.

**How to apply:** For any long-running collection/training scripts, run with a long timeout (600000ms) and no run_in_background flag.
