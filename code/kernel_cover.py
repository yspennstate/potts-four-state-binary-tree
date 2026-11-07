"""Exact Bernstein certificates for Proposition 6.2 (K >= 0), in python-flint.

For each of the 24 relative orders pi, each of the 4 cells of the sorted chart and each endpoint m = 0, m = M, the
kernel is written on the chart z, w >= 0 of the two children's simplices: alpha_i = L rows_i.z, beta_i = L rows_{pi(i)}.w,
P = D U, Q = D V (U = sum z, V = sum w), a = alpha/P, b = beta/Q, s = S/(4PQ) with S = 4PQ + sum alpha beta.
Every bihomogeneous part of bidegree (i, j) in (a, b) becomes num * P^(5-i) Q^(5-j) S^3; the three rational
features use F3 P^5Q^5S^3 = (sum H^3) P^4 Q^4 S / 4, F4 -> (sum H^2)^2 P^4 Q^4 / 4, R -> (sum H^2) C^2 P^2 Q^2 S^2 / 4,
with H_i = 4 alpha_i Q + 4 beta_i P + 4 alpha_i beta_i - C, C = sum alpha beta. So G = K P^5 Q^5 S^3 (UV)^e is a
polynomial with the sign of K on the chart. A case is certified when G - sum_k gamma_k G(SOS_k), gamma >= 0, after
the cell substitution has every coefficient >= 0.

  python kernel_cover.py [delta]    all 192 cases at the shift delta of g_24 (default 1743/480965); writes
                                   results/kernel_cover.jsonl and prints the cases that fail
"""
import itertools, json, sys, time
from fractions import Fraction as Fr
from pathlib import Path
import flint

HERE = Path(__file__).resolve().parent
LOG = HERE / "results" / "kernel_cover.log"
OUT = HERE / "results" / "kernel_cover.jsonl"
L, D = 677000, 3000000
ROWS = ((-3, -3, -3, 0), (-3, -3, 1, 0), (-3, 3, 1, 0), (9, 3, 1, 0))
PERMS = list(itertools.permutations(range(4)))
M = Fr(1375940705703331, 3799912185593856)
BETA = Fr(1, 1000000)
G_BASE = [Fr(x) for x in json.load(open(HERE / "data" / "kernel_coefficients.json", encoding="utf-8"))["g"]]
ctx = flint.fmpq_mpoly_ctx.get(["z0", "z1", "z2", "z3", "w0", "w1", "w2", "w3"], "lex")
GENS = ctx.gens()
Z, Wv = GENS[:4], GENS[4:]


def q(x):
    x = Fr(x); return flint.fmpq(x.numerator, x.denominator)


def say(*a):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a) + "\n")


class Bi:
    """sum over (i, j) of num_ij / (P^i Q^j): a polynomial in (a, b) by bidegree."""
    def __init__(self, parts=None): self.p = parts or {}
    def __add__(self, o):
        r = dict(self.p)
        for k, v in o.p.items():
            r[k] = r[k] + v if k in r else v
        return Bi(r)
    def __neg__(self): return Bi({k: -v for k, v in self.p.items()})
    def __sub__(self, o): return self + (-o)
    def __mul__(self, o):
        if not isinstance(o, Bi):
            return Bi({k: v * q(o) for k, v in self.p.items()})
        r = {}
        for (i, j), u in self.p.items():
            for (k, l), v in o.p.items():
                key = (i + k, j + l); r[key] = r[key] + u * v if key in r else u * v
        return Bi(r)
    __rmul__ = __mul__
    def __pow__(self, n):
        r = Bi({(0, 0): ctx.from_dict({(0,) * 8: 1})})
        for _ in range(n):
            r = r * self
        return r


def build(index, m):
    pi = PERMS[index]
    al = [L * sum(r * x for r, x in zip(ROWS[i], Z)) for i in range(4)]
    be = [L * sum(r * x for r, x in zip(ROWS[pi[i]], Wv)) for i in range(4)]
    U = sum(Z); V = sum(Wv); P = D * U; Qp = D * V
    a = [Bi({(1, 0): x}) for x in al]; b = [Bi({(0, 1): x}) for x in be]
    one = Bi({(0, 0): ctx.from_dict({(0,) * 8: 1})})
    S_ = lambda xs: sum(xs[1:], xs[0])
    A = S_([x * x for x in a]); B = S_([y * y for y in b]); ta = S_([x ** 3 for x in a]); tb = S_([y ** 3 for y in b])
    sa = S_([x ** 4 for x in a]); sb = S_([y ** 4 for y in b]); c = S_([x * y for x, y in zip(a, b)])
    p21 = S_([x * x * y for x, y in zip(a, b)]); p12 = S_([x * y * y for x, y in zip(a, b)]); ell = p21 + p12
    p22 = S_([x * x * y * y for x, y in zip(a, b)]); p31 = S_([x ** 3 * y + x * y ** 3 for x, y in zip(a, b)])
    p32 = S_([x ** 3 * y * y + x * x * y ** 3 for x, y in zip(a, b)]); p33 = S_([x ** 3 * y ** 3 for x, y in zip(a, b)])
    p42 = S_([x ** 4 * y * y + x * x * y ** 4 for x, y in zip(a, b)])
    c4sum = S_([S_([a[i] * b[pp[i]] for i in range(4)]) ** 4 for pp in PERMS])
    t3 = Fr(1, 3); t4 = Fr(1, 4); m = Fr(m)
    zero2 = c * c - A * B * t3; center5 = c * ell - (B * ta + A * tb) * t3
    # the three rational features enter as -R (base), m*F3 (feature 0), -m*F4 (feature 1); kept as separate polys
    feats = [m and None, None, (A + B) * zero2, zero2 * m, center5, c * (A + B), c * (A * A + B * B), B * p21 + A * p12,
             c * c * c - ta * tb * Fr(2, 3), c * A * B, c * (A + B) * m, c * ta * tb, ell * m, (p22 - A * B * t4) * m, p31 * m,
             c * (ta + tb) * m, ell * (A + B) * m, center5 * m, (p32 - (B * ta + A * tb) * t4) * m, c * p22 - ta * tb * t3,
             p33 - ta * tb * t4, p42 - (B * sa + A * sb) * t4, c * (A * A + B * B) * m, c ** 4 * 24 - c4sum,
             A * B * (A + B) - (A * A + B * B) * m, B * ta + A * tb - (ta + tb) * m, (A - one * m) * (B - one * m) * m,
             (A + B) * (A - one * m) * (B - one * m), (A + B - one * (2 * m)) * (m * m), c * p31 - (B * sa + A * sb) * t3,
             p21 * p21 + p12 * p12 - (B * sa + A * sb - A * B * (A + B) * t4) * t3, p21 * p12 - ta * tb * t3,
             (A + B) * (p22 - A * B * t4)]
    sos = [A * B * (A + B) - ta * tb * 2, (A + B) * (A * B - c * c), (A * B - c * c) * m, A * B * (A + B) + ta * tb * 2]
    C = sum(x * y for x, y in zip(al, be)); S = 4 * P * Qp + C
    H = [4 * x * Qp + 4 * y * P + 4 * x * y - C for x, y in zip(al, be)]
    NH = sum(h * h for h in H); JH = sum(h ** 3 for h in H)
    S3 = S ** 3
    Ppow = [P ** k for k in range(6)]; Qpow = [Qp ** k for k in range(6)]

    def lift(bi):
        out = ctx.from_dict({})
        for (i, j), v in bi.p.items():
            assert i <= 5 and j <= 5, (i, j)
            out += v * Ppow[5 - i] * Qpow[5 - j]
        return out * S3
    F3 = JH * P ** 4 * Qp ** 4 * S * q(t4)
    F4 = NH * NH * P ** 4 * Qp ** 4 * q(t4)
    R = NH * C * C * P ** 2 * Qp ** 2 * S ** 2 * q(t4)
    base = lift(ta * tb * Fr(2, 3) - A * B * (BETA * m)) - R
    f0 = q(m) * F3 - lift((ta + tb) * m)
    f1 = lift(A * B * (A + B) + (A * A + B * B) * m) - q(m) * F4
    feat_polys = [f0, f1] + [lift(f) for f in feats[2:]]
    return base, feat_polys, [lift(s) for s in sos], U, V, S


def cell_sub(poly, cell):
    pairs = [(0, 3), (3, 0)]
    first = pairs[cell // 2]; second = pairs[cell % 2]
    sub = list(GENS)
    i, j = first; sub[i] = GENS[i] + GENS[j]
    i, j = second; sub[4 + i] = GENS[4 + i] + GENS[4 + j]
    return poly.compose(*sub)


def kernel_poly(base, feats, g, delta):
    K = base
    for gj, f in zip(g, feats):
        if gj:
            K += q(gj) * f
    return K + q(delta) * feats[24]


def fl(v):
    return float(Fr(int(v.p), int(v.q)))


def negatives(G):
    return sum(1 for v in G.coeffs() if v < 0)


def lp_gamma(K, sos):
    """gamma >= 0 maximising the worst normalised coefficient of K - sum gamma_k sos_k; floats, rationalised later."""
    import numpy as np
    from scipy.optimize import linprog
    dk = K.to_dict(); ds = [s.to_dict() for s in sos]
    keys = set(dk)
    for d in ds:
        keys |= set(d)
    keys = list(keys)
    b = np.array([fl(dk[k]) if k in dk else 0.0 for k in keys])
    Am = np.array([[fl(d[k]) if k in d else 0.0 for d in ds] for k in keys])
    n = np.maximum(np.abs(b), np.abs(Am).max(1)); keep = n > 0
    b = b[keep] / n[keep]; Am = Am[keep] / n[keep, None]
    # maximise s subject to Am gamma + s <= b, gamma >= 0, s <= 1
    A_ub = np.hstack([Am, np.ones((len(b), 1))])
    res = linprog(np.r_[np.zeros(4), -1.0], A_ub=A_ub, b_ub=b, bounds=[(0, None)] * 4 + [(None, 1)], method="highs")
    return (res.x[:4], -res.fun) if res.status == 0 else (None, None)


def solve(delta, elevs=(1, 2)):
    done = set()
    if OUT.exists():
        for line in OUT.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("mode") == "solve" and r["ok"] and Fr(r["delta"]) == delta:
                done.add(tuple(r["case"]))
    say("solve: delta", delta, float(delta), "already done", len(done))
    fails = []
    for end in ("ZERO", "M"):
        m = Fr(0) if end == "ZERO" else M
        for idx in range(24):
            if all((end, cell, idx) in done for cell in range(4)):
                continue
            t0 = time.time(); base, feats, sos, U, V, S = build(idx, m)
            K0 = kernel_poly(base, feats, G_BASE, delta)
            for cell in range(4):
                if (end, cell, idx) in done:
                    continue
                result = None
                for elev in elevs:
                    lift_e = (U * V) ** elev
                    Kc = cell_sub(K0 * lift_e, cell)
                    Sc = [cell_sub(s * lift_e, cell) for s in sos]
                    tried = [[Fr(0)] * 4, [Fr(1, 3), 0, 0, 0]]
                    for gam in tried:
                        G = Kc - sum((q(gk) * s for gk, s in zip(gam, Sc) if gk), ctx.from_dict({}))
                        if negatives(G) == 0:
                            result = (elev, gam, "seed"); break
                    if result:
                        break
                    x, marg = lp_gamma(Kc, Sc)
                    if x is not None and marg > 0:
                        for den in (10 ** 3, 10 ** 6, 10 ** 9, 10 ** 12):
                            gam = [max(Fr(0), Fr(float(v)).limit_denominator(den)) for v in x]
                            G = Kc - sum((q(gk) * s for gk, s in zip(gam, Sc) if gk), ctx.from_dict({}))
                            if negatives(G) == 0:
                                result = (elev, gam, "lp margin %.3g" % marg); break
                    if result:
                        break
                    say("case", end, cell, idx, "deg", 8 + elev, "not certified, lp margin", marg)
                ok = result is not None
                if not ok:
                    fails.append((end, cell, idx))
                say("case", end, cell, idx, "OK" if ok else "FAIL", "" if not ok else "deg %d gamma %s (%s)" % (8 + result[0], [str(g) for g in result[1]], result[2]),
                    round(time.time() - t0, 1), "s")
                with OUT.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(mode="solve", case=[end, cell, idx], delta=str(delta), ok=ok,
                                            degree=None if not ok else 8 + result[0],
                                            gamma=None if not ok else [str(g) for g in result[1]])) + "\n")
    say("solve DONE fails", len(fails), fails)


if __name__ == "__main__":
    delta = Fr(sys.argv[1]) if len(sys.argv) > 1 else Fr(1743, 480965)
    solve(delta)
