"""Coefficient comparisons for Proposition 5.1 (the sign of T at a fixed point), in python-flint.

Built from the definitions of G_C, R_C and the sorted-simplex chart, for all six cosets of the Klein group. For each coset C it builds
N_C = f*(G_C + R_C/2) and Z = f*A*B with f = 48 D^5 P_C^2 U^3 V^3, checks that Z has positive coefficients on the
same support, and reports the exact extremes of coefficient ratios N_C/Z. The claim needs, per coset,
beta = 69763/500000 <= min ratio and max ratio <= M_C, with the six-coset average of the upper constants < 7/6.
Also checks the polynomial identities against direct rational evaluation at random barycentric points.
"""
import itertools, json, random, sys, time
from fractions import Fraction as F
import flint

L, D = 707107, 3000000
ROWS = [(-3, -3, -3, 0), (-3, -3, 1, 0), (-3, 3, 1, 0), (9, 3, 1, 0)]
ctx = flint.fmpz_mpoly_ctx.get((("z", 4), ("w", 4)), "deglex") if False else None
names = ["z0", "z1", "z2", "z3", "w0", "w1", "w2", "w3"]
ctx = flint.fmpz_mpoly_ctx.get(names, "lex")
g = list(ctx.gens())
z, w = g[:4], g[4:]
U, V = sum(z), sum(w)
A = [L * sum(r[k] * z[k] for k in range(4)) for r in ROWS]
B = [L * sum(r[k] * w[k] for k in range(4)) for r in ROWS]
A2 = sum(x * x for x in A); B2 = sum(x * x for x in B)
ta = sum(x ** 3 for x in A); tb = sum(x ** 3 for x in B)

perms = list(itertools.permutations(range(4)))
K = [(0, 1, 2, 3), (1, 0, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0)]
compose = lambda p, q: tuple(p[q[i]] for i in range(4))  # (p o q)(i)
cosets, seen = [], set()
for p in perms:
    if p in seen:
        continue
    c = sorted({compose(p, k) for k in K})
    assert len(c) == 4
    seen.update(c); cosets.append(c)
assert len(cosets) == 6 and len(seen) == 24
for c in cosets:  # right and left cosets agree (K is normal)
    assert sorted({compose(k, c[0]) for k in K}) == c


def build(coset):
    S, Jh, C = [], [], []
    for p in coset:
        Cp = sum(A[i] * B[p[i]] for i in range(4))
        Sp = 4 * D * D * U * V + Cp
        H = [4 * D * (A[i] * V + B[p[i]] * U) + 4 * A[i] * B[p[i]] - Cp for i in range(4)]
        S.append(Sp); Jh.append(sum(h ** 3 for h in H)); C.append(Cp)
    P = S[0] * S[1] * S[2] * S[3]
    P2 = P * P
    term1 = 0
    for j in range(4):
        Pm = 1
        for k in range(4):
            if k != j:
                Pm = Pm * S[k]
        term1 = term1 + Jh[j] * Pm * Pm
    Fpoly = 3 * D ** 3 * U * U * V * V * term1 - 48 * D * D * (ta * V ** 3 + tb * U ** 3) * P2 + 4 * (B2 * ta * V + A2 * tb * U) * P2
    Qpoly = 12 * D * U * V * (3 * sum(x * x for x in C) - 4 * A2 * B2) * P2
    N = 2 * Fpoly + Qpoly          # = 2 f (G + R/2)
    Z = 48 * D * U * V * A2 * B2 * P2   # = f A B
    return N, Z, S


def direct(coset, zz, ww):
    """G_C + R_C/2 and A*B by Bayes-side formulas in exact rationals at a barycentric point."""
    Uz, Vw = sum(zz), sum(ww)
    a = [F(L, D) * sum(r[k] * zz[k] for k in range(4)) / Uz for r in ROWS]
    b = [F(L, D) * sum(r[k] * ww[k] for k in range(4)) / Vw for r in ROWS]
    Aa = sum(x * x for x in a); Bb = sum(x * x for x in b)
    t_a = sum(x ** 3 for x in a); t_b = sum(x ** 3 for x in b)
    Jav, c2 = 0, 0
    for p in coset:
        bp = [b[p[i]] for i in range(4)]
        c = sum(a[i] * bp[i] for i in range(4)); s = 1 + c / 4
        # posterior check: (1+a_i)(1+b_i)/(4s) sums to 1 and 4P-1 = H/s
        post = [(1 + a[i]) * (1 + bp[i]) / (4 * s) for i in range(4)]
        assert sum(post) == 1
        H = [a[i] + bp[i] + a[i] * bp[i] - c / 4 for i in range(4)]
        assert all(4 * post[i] - 1 == H[i] / s for i in range(4))
        Jav += sum(h ** 3 for h in H) / s ** 2 / 4; c2 += c * c / 4
    G = Jav - t_a - t_b + (Bb * t_a + Aa * t_b) / 12
    R = 3 * c2 - Aa * Bb
    return G + R / 2, Aa * Bb


def ev(poly, pt):
    return poly(*pt)


out = {}
random.seed(20261017)
t0 = time.time()
for ci, coset in enumerate(cosets):
    N, Z, S = build(coset)
    nt = dict((tuple(e), int(c)) for e, c in N.terms())
    zt = dict((tuple(e), int(c)) for e, c in Z.terms())
    assert set(nt) == set(zt), "supports differ"
    assert all(c > 0 for c in zt.values())
    for e in nt:
        assert sum(e[:4]) == 11 and sum(e[4:]) == 11
    lo = min(F(nt[e], 2 * zt[e]) for e in nt)
    hi = max(F(nt[e], 2 * zt[e]) for e in nt)
    sneg = sum(1 for p in S for e, c in p.terms() if int(c) <= 0)
    # identity checks at random rational barycentric points (one with a zero weight, one at u for b)
    pts = [([F(random.randint(0, 9)) for _ in range(4)], [F(random.randint(0, 9)) for _ in range(4)]) for _ in range(4)]
    pts.append(([F(1), F(0), F(2), F(5)], [F(0), F(0), F(0), F(1)]))   # b uniform
    pts.append(([F(1), F(0), F(0), F(0)], [F(0), F(1), F(0), F(0)]))   # vertex pair
    pts = [(zz, ww) for zz, ww in pts if sum(zz) > 0 and sum(ww) > 0]
    for zz, ww in pts:
        val, ab = direct(coset, zz, ww)
        pt = [flint.fmpz(int(x)) for x in zz + ww]
        Uz, Vw = sum(zz), sum(ww)
        Pv = 1
        for p in S:
            Pv *= F(int(p(*pt)))
        f = 48 * F(D) ** 5 * Pv * Pv * Uz ** 3 * Vw ** 3
        assert F(int(N(*pt))) == 2 * f * val, ("identity N", ci)
        assert F(int(Z(*pt))) == f * ab, ("identity Z", ci)
    out[ci] = dict(coset=[list(p) for p in coset], terms=len(nt), min_ratio=str(lo), max_ratio=str(hi),
                   min_float=float(lo), max_float=float(hi), nonpositive_S_coefficients=sneg, identity_points=len(pts))
    print(ci, coset, len(nt), float(lo), float(hi), "S<=0 coeffs", sneg, round(time.time() - t0, 1), "s", flush=True)

beta = F(69763, 500000)
ok_lo = all(F(v["min_ratio"]) >= beta for v in out.values())
avg_hi = sum(F(v["max_ratio"]) for v in out.values()) / 6
print("all six min ratios >= 69763/500000:", ok_lo)
print("average of six max ratios:", float(avg_hi), "< 7/6:", avg_hi < F(7, 6))
json.dump(dict(cosets=out, beta_ok=ok_lo, avg_upper=str(avg_hi), avg_upper_lt_7_6=avg_hi < F(7, 6)),
          open(__import__("pathlib").Path(__file__).resolve().parent / "results" / "cubic_gain.json", "w"), indent=1)
