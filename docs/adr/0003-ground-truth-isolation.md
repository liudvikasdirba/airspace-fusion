---
title: "0003: Keep ground truth away from the tracker"
status: accepted
updated: 2026-10-31
---

# 0003: Keep ground truth away from the tracker

## Context

A tracker that can see ground truth, even by accident through shared objects, produces results that look
better than they are.

## Decision

The simulator keeps truth and truth-side labels apart from the emitted records. The tracker (`estimation.py`)
receives only validated measurements and the scan schedule. Truth and tracker output meet only in
`evaluation.py`, and in the dashboard's plotting. The learning loop uses truth only to score settings; it
never passes truth to the tracker.

## Consequences

The module dependency rules in `tests/test_integration.py` enforce this: `estimation` may not import
`simulator`. Each new module needs a rule there. The dashboard's "false observation" markers use truth-side
labels for display only.
