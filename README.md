# snake-laya — human vs Laya snake

Two side-by-side boards: you steer with arrow keys, the computer's snake is steered by [Laya](https://github.com/NandhaKishorM/laya) (one `choice` decision per step), with live decision telemetry. Design: `../00draft/snake-laya-design.md`.

## Setup

```bash
uv sync
```

- Python 3.12 (`.python-version`); deps managed by `uv` only
- First run downloads the checkpoint from Hugging Face; cached afterwards

## Run

```bash
uv run snake-laya                                  # sync versus, english checkpoint, auto device
uv run snake-laya --mode max                       # unranked showcase: computer steps per decision
uv run snake-laya --model multilingual --device mps
uv run snake-laya --log run.jsonl                  # one JSONL record per computer step
uv run snake-laya --bench 400 --device mps         # headless throughput/quality, no display needed
uv run snake-laya --brain heuristic                # no model: heuristic baseline plays
```

| Flag | Values | Default |
|---|---|---|
| `--mode` | `sync` (ranked, computer steps on the tick), `max` (unranked showcase) | `sync` |
| `--tick-ms` | int > 0 | `120` |
| `--grid` | `WxH`, min `12x8` | `30x20` |
| `--duration` | active seconds > 0 | `180` |
| `--seed` | int | random, shown in footer |
| `--model` | `english`, `multilingual`, `typed-decisions` (experimental, warns) | `english` |
| `--device` | `cpu`, `cuda`, `mps` | auto |
| `--brain` | `laya`, `heuristic` | `laya` |
| `--no-safety` | execute Laya's raw pick even if fatal | off |
| `--log PATH` | append JSONL decision log | off |
| `--bench N` | headless: N decisions, print stats, exit | off |

Invalid values exit with code 2.

## Controls

| Key | Effect |
|---|---|
| Arrows | queue a turn (≤2 buffered, one per tick; reverse/duplicate dropped) |
| `Space` | pause/resume (freezes clock, both boards, respawn timers, inference requests) |
| `R` | restart match (same seed) |
| `M` | switch mode → `Y` confirm (restarts match) / `N`·`Esc` cancel |
| `Esc` | quit |

## Rules

- Score +1 per food; death → respawn after 1 s of active time, score resets, `TOTAL` keeps counting
- Winner (`sync` only): total food, then best life, then fewer deaths
- Same seed = same initial board; food diverges once bodies differ

## Telemetry (computer panel)

| Metric | Meaning |
|---|---|
| `PREDICT` / `P50` / `P95` | `agent.predict` wall time: last / rolling 200 |
| `DECISIONS/S` | applied decisions in the last active second |
| `DECISIONS` | predictions applied to a step |
| prob bars | Laya probabilities over offered moves; reverse is never offered (`—`) |
| `SHARPNESS` | Laya `confidence` = 1 − H(p)/log k; distribution sharpness, **not** success probability |
| `OVERRIDES` | fatal pick replaced by the most probable safe move |
| `LATE` | ticks executed without a current prediction (continue straight, safety-checked) |
| `BASELINE AGREE` | raw pick == heuristic baseline (a heuristic, not an oracle) |

## Measured (Apple MPS, seed 1, 400 decisions, `--bench`)

| Brain | P50 ms | decisions/s | food | overrides |
|---|---|---|---|---|
| `english` | 40 | 25.5 | 20 | 0.8% |
| `multilingual` | 20 | 46.6 | 2 | 21.2% |
| `heuristic` | ~0 | 520 | 19 | 0% |

`english` P50 fits the 120 ms tick with margin (GUI adds ~15 ms contention).

## Code map (`src/snake_laya/`)

| Module | Purpose |
|---|---|
| `game.py` | pure rules: `Board`, `Dir`, collision, seeded row-major food |
| `features.py` | per-direction facts (fatal, food distance, flood-fill room, trap) → Laya state/question text |
| `brain.py` | `LayaBrain`, `HeuristicBrain`, `apply_safety`, `late_fallback` |
| `inference.py` | `InferenceWorker`: one predict at a time, latest-request-only mailbox |
| `runner.py` | `ComputerRunner`: owns computer board, sync deadline / max pacing, generation-matched results |
| `match.py` | phases, `HumanController`, `InputQueue`, confirm/reset/pause |
| `clock.py` | `ActiveClock`: pause-aware time shared by all components |
| `stats.py` | `DecisionStats`, `HumanStats`, `rank` |
| `decision_log.py` | line-buffered JSONL log |
| `ui.py` | pygame renderer |
| `main.py` | CLI, `--bench`, GUI loop, ordered shutdown |

## Tests

```bash
uv run pytest                                          # fast suite; real-model tests skipped with reason
uv run pytest --run-slow tests/test_laya_smoke.py      # loads real english + multilingual checkpoints
uvx ruff check src tests && uvx ruff format --check src tests
```

## Known limits

- Laya base checkpoints are weak zero-shot at spatial reasoning: all geometry is precomputed into the option text. Prompt wording matters a lot (tuning notes in `features.py` docstring)
- `multilingual` is ~2× faster but near-random on this prompt; `english` is the default for that reason
- CPU inference (~200–500 ms per Laya README) exceeds the default tick: expect high `LATE`; raise `--tick-ms` or use `--device mps/cuda`
- Accelerator op errors during inference are not retried on CPU: the error overlay suggests `--device cpu`
- `--log` output is raw telemetry (input, decision, outcome), not a ready training set
