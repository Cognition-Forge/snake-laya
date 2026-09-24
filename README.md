# snake-laya — human vs Laya snake

Two side-by-side boards: you steer with arrow keys, the computer's snake is steered by [Laya](https://github.com/NandhaKishorM/laya) (one `choice` decision per step), with live decision telemetry. Design: `../00draft/snake-laya-design.md`.

Two model runtimes serve the same three checkpoints: `--brain laya` (upstream PyTorch) and `--brain laya-mlx` ([laya-mlx](https://github.com/mizorewww/laya-mlx), Apple silicon). They pick identically; MLX is ~1.2–1.6× faster here.

## Setup

```bash
uv sync                 # laya (PyTorch) + heuristic
uv sync --extra mlx     # also laya-mlx; Apple silicon only
```

- Python 3.12 (`.python-version`); deps managed by `uv` only
- First run of a checkpoint downloads only that checkpoint (pinned revision in `brain.py`) to the Hugging Face cache (`~/.cache/huggingface/hub`, `HF_HOME` to move); later starts load from cache with no network
- Each backend has its own weights: `laya` pins one bundled repo (`LAYA_REVISION`), `laya-mlx` pins one repo per checkpoint (`LAYA_MLX_MODELS`). Switching backends downloads once more
- New upstream revision: bump the pin in the matching `Checkpoint`; the next start downloads it once

## Run

```bash
uv run snake-laya                                  # sync versus, english checkpoint, auto device
uv run snake-laya --mode max                       # unranked showcase: computer steps per decision
uv run snake-laya --computer-tick-ms 60            # computer steps 2× faster than you (unranked)
uv run snake-laya --model multilingual --device mps
uv run snake-laya --brain laya-mlx                 # same weights, MLX runtime (Apple silicon)
uv run snake-laya --brain laya-mlx --device cpu    # MLX on CPU instead of the Metal GPU
uv run snake-laya --log run.jsonl                  # one JSONL record per computer step
uv run snake-laya --bench 400 --device mps         # headless throughput/quality, no display needed
uv run snake-laya --brain heuristic                # no model: heuristic baseline plays
```

| Flag | Values | Default |
|---|---|---|
| `--mode` | `sync` (ranked, computer steps on the tick), `max` (unranked showcase) | `sync` |
| `--tick-ms` | int > 0 | `120` |
| `--computer-tick-ms` | int > 0; computer step interval in `sync` (ignored in `max`); ≠ `--tick-ms` → unranked | `--tick-ms` |
| `--grid` | `WxH`, min `12x8` | `30x20` |
| `--duration` | active seconds > 0 | `180` |
| `--seed` | int | random, shown in footer |
| `--model` | `english`, `multilingual`, `typed-decisions` (experimental, warns) | `english` |
| `--device` | `laya`: `cpu`, `cuda`, `mps` · `laya-mlx`: `cpu`, `gpu`, `metal` | auto |
| `--brain` | `laya` (PyTorch), `laya-mlx` (Apple silicon), `heuristic` | `laya` |
| `--no-safety` | execute Laya's raw pick even if fatal | off |
| `--log PATH` | append JSONL decision log | off |
| `--bench N` | headless: N decisions, print stats, exit | off |

Invalid values exit with code 2, including a `--device` the chosen `--brain` does not support.

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
- Winner (`sync` with equal ticks only): total food, then best life, then fewer deaths
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

## Measured (Apple silicon, seed 1, 400 decisions, `--bench`)

| Brain | Model | P50 ms | decisions/s | food | overrides |
|---|---|---|---|---|---|
| `laya` (mps) | `english` | 39 | 25.3 | 20 | 0.8% |
| `laya` (mps) | `multilingual` | 20 | 46.6 | 2 | 21.2% |
| `laya-mlx` (gpu) | `english` | 32 | 33.2 | 20 | 0.8% |
| `laya-mlx` (gpu) | `multilingual` | 13 | 67.6 | 2 | 21.2% |
| `heuristic` | — | ~0 | 520 | ~19 | 0% |

`english` P50 fits the 120 ms tick with margin on both backends (GUI adds ~15 ms contention).

Both backends played the identical game on each checkpoint (same food, deaths and override rate; mean sharpness 0.158 vs 0.159), so the MLX port is a drop-in swap for speed, not a behaviour change.

## Code map (`src/snake_laya/`)

| Module | Purpose |
|---|---|
| `game.py` | pure rules: `Board`, `Dir`, collision, seeded row-major food |
| `features.py` | per-direction facts (fatal, food distance, flood-fill room, trap) → Laya state/question text |
| `brain.py` | `LayaBrain` (either backend), `HeuristicBrain`, `Checkpoint` pins, `apply_safety`, `late_fallback`, `resolve_checkpoint` (cache-first, pinned revision) |
| `inference.py` | `InferenceWorker`: one predict at a time, latest-request-only mailbox |
| `runner.py` | `ComputerRunner`: owns computer board, sync deadline (computer tick) / max pacing, generation-matched results |
| `match.py` | phases, `HumanController`, `InputQueue`, confirm/reset/pause, ranked/unranked |
| `clock.py` | `ActiveClock`: pause-aware time shared by all components |
| `stats.py` | `DecisionStats`, `HumanStats`, `rank` |
| `decision_log.py` | line-buffered JSONL log |
| `ui.py` | pygame renderer |
| `main.py` | CLI, `--bench`, GUI loop, ordered shutdown |

## Tests

```bash
uv run pytest                                          # fast suite; real-model tests skipped with reason
uv run pytest --run-slow tests/test_laya_smoke.py      # real english + multilingual on every installed backend
uvx ruff check src tests && uvx ruff format --check src tests
```

## Known limits

- Laya base checkpoints are weak zero-shot at spatial reasoning: all geometry is precomputed into the option text. Prompt wording matters a lot (tuning notes in `features.py` docstring)
- `multilingual` is ~2× faster but near-random on this prompt; `english` is the default for that reason
- CPU inference (~200–500 ms per Laya README) exceeds the default tick: expect high `LATE`; raise `--tick-ms` or use an accelerator
- Accelerator op errors during inference are not retried on CPU: the error overlay suggests `--device cpu`
- `laya-mlx` needs Apple silicon (`mlx` ships arm64-macOS wheels only) and is an independent port, not an official Convai release; it is not installed by a plain `uv sync`
- `laya-mlx` warns at load that it clamps an out-of-range calibration temperature. That bucket is for questions with 11+ options, which this 3-move prompt never reaches, so `SHARPNESS` is unaffected
- `--log` output is raw telemetry (input, decision, outcome), not a ready training set
