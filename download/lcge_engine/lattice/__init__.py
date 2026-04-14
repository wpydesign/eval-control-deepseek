"""
lattice/ — LCGE v1.3 Behavioral Coordinate System

This package defines the empirical coordinate system of the behavioral
response manifold BEFORE any scoring or classification exists.

Structure:
    frozen_prompts.py      — fixed, non-generated evaluation prompts
    variant_generator.py   — one variant per perturbation strategy
    run_lattice.py         — evaluation matrix runner (N repetitions)
    vector_store.py        — raw response → embedding vectors
    coordinate_solver.py   — distance matrix + PCA/SVD decomposition

Output:
    output/behavioral_space.json — the discovered coordinate system

Hard constraint: NO classification, NO instability types, NO triggers,
NO scoring weights, NO "dominant mode". Only empirical geometry.
"""

__version__ = "1.3.0-pre"
