#!/usr/bin/env python3
"""
validate_v13.py — Validates the v1.3 lattice infrastructure.

Tests all components with synthetic data (no API calls needed):
1. Frozen prompts loaded correctly
2. Lattice index generation (35 points = 5 prompts × 7 strategies)
3. Variant generator produces one variant per strategy
4. Vector store builds embedding matrix
5. Coordinate solver produces PCA decomposition
6. Output format matches spec
"""

import sys
import os
import json
import numpy as np

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_engine_dir = os.path.dirname(_pkg_dir)
_parent_dir = os.path.dirname(_engine_dir)
for d in [_parent_dir, _engine_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_frozen_prompts():
    section("TEST 1: Frozen prompts")
    from lcge_engine.lattice.frozen_prompts import FROZEN_PROMPTS
    assert len(FROZEN_PROMPTS) == 5
    assert "What is the capital of France?" in FROZEN_PROMPTS
    assert "What is 2 + 2?" in FROZEN_PROMPTS
    assert "Explain how quicksort works." in FROZEN_PROMPTS
    assert "Is it ethical to lie to protect someone?" in FROZEN_PROMPTS
    assert "Write a short summary of climate change." in FROZEN_PROMPTS
    print(f"  {len(FROZEN_PROMPTS)} frozen prompts loaded")
    print("  PASS")


def test_lattice_index():
    section("TEST 2: Lattice index generation")
    from lcge_engine.lattice.frozen_prompts import FROZEN_PROMPTS
    from lcge_engine.lattice.variant_generator import generate_lattice_index, STRATEGY_NAMES

    lattice = generate_lattice_index(FROZEN_PROMPTS)
    expected = len(FROZEN_PROMPTS) * len(STRATEGY_NAMES)
    assert len(lattice) == expected, f"Expected {expected}, got {len(lattice)}"

    # Each prompt has all strategies
    for pid in range(len(FROZEN_PROMPTS)):
        point_strategies = set(l['strategy'] for l in lattice if l['prompt_id'] == pid)
        assert point_strategies == set(STRATEGY_NAMES)

    # No classification artifacts
    for point in lattice:
        assert 'instability_type' not in point
        assert 'score' not in point
        assert 'trigger' not in point

    print(f"  {len(lattice)} lattice points ({len(FROZEN_PROMPTS)} prompts × {len(STRATEGY_NAMES)} strategies)")
    print("  No classification artifacts present")
    print("  PASS")


def test_variant_generator():
    section("TEST 3: Variant generator")
    from lcge_engine.lattice.variant_generator import generate_strategy_variants

    variants = generate_strategy_variants("What is 2+2?")
    assert len(variants) == 7
    strategies = [v['strategy'] for v in variants]
    expected_strategies = [
        'paraphrase', 'constraint_add', 'constraint_remove',
        'role_change', 'format_change', 'step_by_step', 'adversarial'
    ]
    assert strategies == expected_strategies

    # All variants contain the seed prompt
    for v in variants:
        assert "What is 2+2?" in v['prompt']

    print(f"  7 variants generated (one per strategy)")
    print("  PASS")


def test_vector_store():
    section("TEST 4: Vector store")
    from lcge_engine.lattice.vector_store import load_run_records, build_embedding_matrix

    # Generate synthetic records
    records = []
    for i in range(50):
        records.append({
            'run_id': f'test_{i}',
            'response': f'This is synthetic response number {i} with some varied content about topic A.',
            'row_index': i,
        })

    jsonl_path = '/tmp/test_v13_vectors.jsonl'
    with open(jsonl_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')

    loaded = load_run_records(jsonl_path)
    assert len(loaded) == 50

    embeddings, meta = build_embedding_matrix(loaded)
    assert embeddings.shape[0] == 50
    assert embeddings.shape[1] > 0
    assert meta['n_features'] > 0

    # Handle empty responses gracefully
    records_with_empty = records + [{'run_id': 'empty_1', 'response': '[ERROR] test', 'row_index': 50}]
    with open(jsonl_path, 'w') as f:
        for r in records_with_empty:
            f.write(json.dumps(r) + '\n')

    loaded2 = load_run_records(jsonl_path)
    embeddings2, meta2 = build_embedding_matrix(loaded2)
    assert embeddings2.shape[0] == 51

    print(f"  Embedding matrix: {embeddings.shape}")
    print(f"  Empty handling: {embeddings2.shape}")
    print("  PASS")


def test_coordinate_solver():
    section("TEST 5: Coordinate solver")
    from lcge_engine.lattice.coordinate_solver import (
        compute_distance_matrix,
        compute_pca,
        compute_strategy_displacement,
    )

    # Build synthetic embedding matrix (strategy-separated clusters)
    np.random.seed(42)
    n_per_strategy = 15
    n_strategies = 7

    # Create cluster centers
    centers = np.random.randn(n_strategies, 20) * 2

    # Generate samples around centers
    all_embeddings = []
    records = []
    strategies = ['paraphrase', 'constraint_add', 'constraint_remove',
                  'role_change', 'format_change', 'step_by_step', 'adversarial']

    for s_idx, strategy in enumerate(strategies):
        for r_idx in range(n_per_strategy):
            sample = centers[s_idx] + np.random.randn(20) * 0.5
            all_embeddings.append(sample)
            records.append({
                'run_id': f'test_{s_idx}_{r_idx}',
                'strategy': strategy,
                'row_index': len(records),
            })

    embeddings = np.array(all_embeddings)

    # Distance matrix
    dist = compute_distance_matrix(embeddings)
    assert dist.shape == (len(records), len(records))
    assert np.allclose(np.diag(dist), 0)

    # PCA
    pca_result = compute_pca(embeddings, n_components=8)
    assert len(pca_result['principal_components']) > 0
    assert pca_result['projections'].shape[0] == len(records)

    # Check variance ratios sum to ~1.0
    total_var = sum(pc['variance_ratio'] for pc in pca_result['principal_components'])
    assert 0.9 < total_var <= 1.01, f"Variance ratios sum to {total_var}"

    # Strategy displacement
    displacements = compute_strategy_displacement(records, pca_result['projections'])
    assert len(displacements) == n_strategies
    for s, d in displacements.items():
        assert 'displacement_from_global' in d
        assert 'n_samples' in d
        assert d['n_samples'] == n_per_strategy

    print(f"  Distance matrix: {dist.shape}")
    print(f"  PCA components: {len(pca_result['principal_components'])}")
    print(f"  Total variance captured: {total_var:.4f}")
    print(f"  Strategy displacements: {len(displacements)}")

    # Check that different strategies have different displacements
    disps = [d['displacement_from_global'] for d in displacements.values()]
    disp_range = max(disps) - min(disps)
    print(f"  Displacement range: {disp_range:.4f}")

    # No classification artifacts in output
    assert 'instability_type' not in displacements
    assert 'dominant_mode' not in displacements
    print("  No classification artifacts")
    print("  PASS")


def test_full_pipeline():
    section("TEST 6: Full pipeline (lattice → vector → solver)")
    from lcge_engine.lattice.coordinate_solver import solve_behavioral_space
    from lcge_engine.lattice.frozen_prompts import FROZEN_PROMPTS
    from lcge_engine.lattice.variant_generator import STRATEGY_NAMES

    # Generate synthetic data (35 lattice points × 3 reps = 105)
    records = []
    for pid in range(len(FROZEN_PROMPTS)):
        for strat in STRATEGY_NAMES:
            for rep in range(3):
                response = f'Synthetic response for prompt {pid} with strategy {strat} repetition {rep}.'
                records.append({
                    'prompt_id': pid,
                    'strategy': strat,
                    'rep': rep,
                    'run_id': f'p{pid}_{strat}_r{rep:03d}',
                    'response': response,
                    'row_index': len(records),
                })

    jsonl_path = '/tmp/test_v13_full.jsonl'
    with open(jsonl_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')

    output_path = '/tmp/test_v13_behavioral_space.json'
    result_path = solve_behavioral_space(
        jsonl_path,
        n_components=8,
        output_path=output_path,
    )

    assert os.path.exists(result_path)
    with open(result_path) as f:
        result = json.load(f)

    # Verify output format matches spec
    assert 'principal_components' in result
    assert 'projections' in result
    assert len(result['projections']) == 105

    for pc in result['principal_components']:
        assert 'variance_ratio' in pc
        assert 'cumulative_variance' in pc

    # Verify NO classification artifacts
    assert 'instability_type' not in result
    assert 'dominant_failure_mode' not in result
    assert 'trigger' not in result
    assert 'component_scores' not in result
    assert 'score' not in result

    # Check auxiliary files created
    assert os.path.exists('/tmp/test_v13_behavioral_space.json')
    output_dir = os.path.dirname(result_path)
    assert os.path.exists(os.path.join(output_dir, 'embeddings.npz'))
    assert os.path.exists(os.path.join(output_dir, 'distance_matrix.npz'))
    assert os.path.exists(os.path.join(output_dir, 'record_index.json'))

    print(f"  Output: {result_path}")
    print(f"  Projections: {len(result['projections'])} records")
    print(f"  PCs: {len(result['principal_components'])}")
    print(f"  Distance summary: {result['distance_matrix_summary']}")
    print(f"  Strategy displacements: {len(result['strategy_displacements'])}")
    print(f"  Auxiliary files: embeddings.npz, distance_matrix.npz, record_index.json")
    print("  NO classification artifacts detected")
    print("  PASS")


def main():
    print("=" * 60)
    print("  LCGE v1.3 Validation — Lattice Infrastructure")
    print("=" * 60)

    tests = [
        test_frozen_prompts,
        test_lattice_index,
        test_variant_generator,
        test_vector_store,
        test_coordinate_solver,
        test_full_pipeline,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
