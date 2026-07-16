# M4 VLM Evidence-Fusion Repair Plan

> **For agentic workers:** Execute this plan task-by-task with test-first checkpoints.

**Goal:** Repair the M4 VLM verifier so that it uses the dense SAM2 track evidence, preserves strong rule evidence when the VLM is uncertain, and is evaluated on the frozen smoke dev10/eval10 split without changing D0, D1, or B1.

**Scope:** M4 only. The active 300-sample D0/D1/B1 pilot remains untouched. M4 configuration selection is limited to dev10; eval10 is run once with the selected configuration.

## Task 1: Freeze the repair contract

- [x] Record the approved behavior: SAM2 trajectory evidence in verifier prompts; object/category/time grouping; confirmed/rejected/uncertain response status; detector-weight sweep 0.7/0.6/0.5; dev10 selection then one eval10 check.
- [x] Keep existing JSON fields and API compatibility where possible; make new response status optional with a safe default for old providers.

## Task 2: Add verifier regression tests first

- [x] Add a failing test proving grouped verification separates candidates by object/category/time segment.
- [x] Add a failing test proving the verifier payload contains serialized SAM2 trajectory/box evidence and an explicit expected-event caution.
- [x] Add a failing test proving an optional `claim_status` response is retained in `VLMReview`.

## Task 3: Implement evidence-aware verification

- [x] Serialize bounded, chronological `FrameState`/track evidence from each candidate's `evidence` field without sending full raw observations.
- [x] Group candidates by object, category, and temporal segment; select representative frames per group.
- [x] Update the verifier schema/prompt to distinguish confirmed, rejected, and uncertain claims, while preserving old-provider compatibility.

## Task 4: Make M4 fusion conservative toward strong detector evidence

- [x] Add a failing test for the M4 default detector weight and explicit CLI/config propagation.
- [x] Change the M4 default detector weight to 0.7; retain `--m4-detector-weight` for the frozen 0.7/0.6/0.5 dev sweep.
- [x] Record the optional status in violation evidence without changing D0/D1/B1 fusion.

## Task 5: Verify locally and remotely

- [x] Run focused red/green tests after each behavior change, then the full local suite.
- [ ] Transfer only the committed M4 source/plan to the remote source tree; do not restart the active pilot.
- [ ] After the pilot frees the endpoint, run independent M4 dev10 weights 0.7/0.6/0.5 using cached SAM2 observations, select one by frozen dev criteria, then run eval10 once.
- [ ] Save configs, predictions, summaries, failure/latency logs, and the outcome in the main full-evaluation plan.

## Execution results

Results are appended after each checkpoint.

### E1 — Evidence-aware M4 implementation

- The grouped verifier now makes one call per `(object, category, start_frame, end_frame)` segment instead of reusing a category-level review across unrelated tracks.
- The pipeline attaches a bounded chronological `sam2_track` snapshot (state count, visible count, frame range, boxes, centers and motion fields) to every rule candidate before VLM review. The payload remains capped at 24 states per track.
- The verifier prompt now explicitly uses SAM2 evidence, distinguishes `confirmed`, `rejected`, and `uncertain`, and states that prompt-expected events must not be rejected by default. Old providers remain valid because `claim_status` is optional and defaults to `uncertain`.
- M4's default detector/VLM fusion is now `0.7/0.3`; `--m4-detector-weight` remains available for the frozen `0.7/0.6/0.5` dev sweep. D0, D1 and B1 code paths are unchanged.
- Focused red/green checks passed, followed by the local full suite: `177 passed in 4.95s` using the repository's ignored basetemp.
- The active remote pilot remains untouched: at checkpoint time it had 175 cached SAM2 observations and 527 predictions, with the original D0/D1/B1 command still running.
