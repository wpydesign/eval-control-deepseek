---
Task ID: v13-1
Agent: Main Agent
Task: LCGE v1.3 — Behavioral Coordinate System Foundation

Work Log:
- Created lattice package: lcge_engine/lattice/ with 6 modules
- frozen_prompts.py: 5 fixed evaluation prompts (factual ×2, reasoning, ethical, instruction)
- variant_generator.py: one variant per strategy (7 strategies), deterministic template selection
- run_lattice.py: evaluation matrix runner — N reps × 35 lattice points, JSONL output, resume support, rate limit retry (3 attempts, 5/10/20s backoff), inter-call delay, crash-safe flush
- vector_store.py: raw response → TF-IDF embedding matrix, handles empty/error responses
- coordinate_solver.py: pairwise cosine distance matrix + PCA decomposition + strategy displacement analysis
- Updated llm_bridge.mjs: added temperature parameter for stochastic sampling (v1.3)
- validate_v13.py: 6 synthetic tests — all pass
- Real LLM execution blocked by persistent API 429 rate limit

Stage Summary:
- Lattice infrastructure complete and validated (6/6 tests pass)
- 35 lattice points = 5 prompts × 7 strategies
- Full pipeline: runs.jsonl → embeddings.npz → distance_matrix.npz → behavioral_space.json
- Output format: principal_components (variance_ratio, cumulative_variance), projections per run_id, strategy_displacements, distance_matrix_summary
- NO classification artifacts anywhere in the lattice pipeline
- Real data collection requires API rate limit to clear
- To run real lattice: python3 lcge_engine/lattice/run_lattice.py --reps 20 --delay 2 --output lattice/output/runs.jsonl
- To solve: python3 -m lcge_engine.lattice.coordinate_solver --input lattice/output/runs.jsonl --output lattice/output/behavioral_space.json
