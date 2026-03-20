---
name: feedback_separate_compute_plot
description: Always separate time-intensive operations (like df.apply) from plotting cells in notebooks
type: feedback
---

Always separate time-intensive operations (like `df.apply`, model inference, embeddings) from plotting code into different notebook cells.

**Why:** Plotting takes a few seconds to tweak and re-run; re-running the compute every time is wasteful.
**How to apply:** When adding analysis to notebooks, put the compute (apply, encoding, etc.) in one cell and the visualization in the next cell.
