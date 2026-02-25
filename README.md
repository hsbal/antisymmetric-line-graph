# Antisymmetric Line Graph (ALG) — Reproducibility Code

This repository contains code to reproduce computations in the paper:

**Antisymmetric Graph Lifts**  
Hartosh Singh Bal

The core construction is the **antisymmetric line graph** `A(G)` (denoted `ALG(G)` in code),
a canonical signed graph on the edge set of an ordinary graph `G`.

## What’s in this repo

- `src/alg.py`  
  Core implementation of:
  - antisymmetric line graph signed adjacency (rule-based and lift-based)
  - signed spectrum and Δ3 = tr(M^3)/6
  - frustration index ℓ(A(G)) (exact with per-instance timeout)
  - MaxCut defect on G (exact for n ≤ 7)
  - NetworkX atlas enumeration of connected graphs up to 7 vertices

- `scripts/`  
  One-command reproduction scripts used in the paper.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
