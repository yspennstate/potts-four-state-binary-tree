# The Kesten–Stigum bound is tight for the four-state Potts model on the binary tree

Yitzchak Shmalo

This repository contains the paper and the programs behind its computer-assisted steps.

A uniformly random colour is broadcast from the root of the infinite binary tree; each edge copies its parent's
colour with probability $\lambda+(1-\lambda)/4$ and otherwise picks one of the other three colours uniformly. For
$0<\lambda\le 1/\sqrt2$ the colours at depth $n$ carry asymptotically no information about the root. Together with
the Kesten–Stigum theorem this makes $2\lambda^2=1$ the reconstruction threshold for the ferromagnetic four-state
Potts channel on the binary tree.

The three-state case on every regular tree is in a separate repository,
[potts-kesten-stigum-small-degree](https://github.com/yspennstate/potts-kesten-stigum-small-degree).

## Contents

- `paper/main.tex`, `paper/main.pdf`: the paper.
- `code/`: the programs for the computer-assisted steps, with their input data in `code/data/` and the outputs of
  the runs used for the paper in `code/results/`.

## The checks

| Result in the paper | What is checked | Program | Recorded output |
|---|---|---|---|
| Proposition 3.1 | the grid upper bound $x_n\le U$ for $n\ge500$ at $\lambda^*=707107/10^6$: 478 orbits, 114481 pairs per step, masses as integers over $2^{48}$; every split is checked to be an exact barycenter and the 500 depths are compared with the certificate | `grid_replay.py` (reads `data/grid_certificate_N36.json`) | `results/grid_replay_recorded.json` |
| Lemma 4.2 | fixed-point messages lie in $\rho(4\Delta-1)$ with $\rho=677/1000$ | `support_check.py` | (prints) |
| Proposition 5.1 | the coefficient comparisons for the sign of $T$, all six Klein cosets, $6\times129600$ inequalities | `cubic_gain_check.py` | `results/cubic_gain_recorded.json` |
| Proposition 6.1, Table 1 | the expectation identities, in exact arithmetic on random symmetric laws | `kernel_expectations_check.py` | (prints) |
| Proposition 6.2 | $K\ge0$: 192 Bernstein certificates (24 relative orders, 4 cells, 2 endpoints) at the shift $\delta=1743/480965$ of $g_{24}$ | `kernel_cover.py` (reads `data/kernel_coefficients.json`) | `results/kernel_cover_recorded.jsonl` |

All arithmetic in the bounds is exact (Python integers and fractions, and `python-flint` polynomials over
$\mathbb{Q}$). The kernel cover uses a floating-point linear program only to propose the weights $\gamma$; every
certificate is then checked exactly.

## Running the checks

Python 3.10 or later with `python-flint` (and `numpy` for the kernel cover):

```
pip install python-flint numpy
cd code
python support_check.py
python kernel_expectations_check.py
python cubic_gain_check.py
python grid_replay.py        # about 20 minutes
python kernel_cover.py       # about an hour
```

`grid_replay.py` and `kernel_cover.py` write their logs and outputs to `code/results/`.
