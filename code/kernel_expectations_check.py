"""Exact check of the expectation identities behind Proposition 6.1 and Table 1.

On random finite S4-invariant zero-sum laws (a few vectors, symmetrized over all 24 colour permutations, random
rational weights), with a and b independent copies, compute in exact rationals:
  - every correction feature j = 2..32 of Table 2 has expectation 0 (features 0 and 1 need the BP
    fixed-point relations F3 = 2sqrt2 T, F4 = 4V and are checked symbolically instead);
  - the drift identity E[N/s]/16 = m/8 - mT/48 - T^2/384 + R/256, R = E[N c^2 / s], with the observation weight s;
  - the entries of Table 1.
Feature numbering follows Table 2; the base is (2/3) ta tb - N c^2/s - beta m A B.
"""
import itertools, random, sys
from fractions import Fraction as F

PERMS = list(itertools.permutations(range(4)))


def law(k, rng):
    pts = []
    for _ in range(k):
        r = F(rng.randint(1, 60), 100)
        w = [F(rng.randint(0, 9)) for _ in range(4)]
        if sum(w) == 0:
            w[0] = F(1)
        v = [r * (4 * x / sum(w) - 1) for x in w]   # zero sum, inside r*(4 Delta - 1)
        wt = F(rng.randint(1, 9))
        for p in PERMS:
            pts.append(([v[p[i]] for i in range(4)], wt))
    tot = sum(wt for _, wt in pts)
    return [(v, wt / tot) for v, wt in pts]


def feats(a, b, m):
    A = sum(t * t for t in a); B = sum(t * t for t in b); ta = sum(t ** 3 for t in a); tb = sum(t ** 3 for t in b)
    sa = sum(t ** 4 for t in a); sb = sum(t ** 4 for t in b); c = sum(x * y for x, y in zip(a, b))
    p21 = sum(x * x * y for x, y in zip(a, b)); p12 = sum(x * y * y for x, y in zip(a, b)); ell = p21 + p12
    p22 = sum(x * x * y * y for x, y in zip(a, b)); p31 = sum(x ** 3 * y + x * y ** 3 for x, y in zip(a, b))
    p32 = sum(x ** 3 * y * y + x * x * y ** 3 for x, y in zip(a, b)); p33 = sum(x ** 3 * y ** 3 for x, y in zip(a, b))
    p42 = sum(x ** 4 * y * y + x * x * y ** 4 for x, y in zip(a, b))
    c4sum = sum(sum(a[i] * b[p[i]] for i in range(4)) ** 4 for p in PERMS)
    zero2 = c * c - A * B / 3; center5 = c * ell - (B * ta + A * tb) / 3
    cs = [None, None, (A + B) * zero2, m * zero2, center5, c * (A + B), c * (A * A + B * B), B * p21 + A * p12,
          c ** 3 - 2 * ta * tb / 3, c * A * B, m * c * (A + B), c * ta * tb, m * ell, m * (p22 - A * B / 4), m * p31,
          m * c * (ta + tb), m * ell * (A + B), m * center5, m * (p32 - (B * ta + A * tb) / 4), c * p22 - ta * tb / 3,
          p33 - ta * tb / 4, p42 - (B * sa + A * sb) / 4, m * c * (A * A + B * B), 24 * c ** 4 - c4sum,
          A * B * (A + B) - m * (A * A + B * B), B * ta + A * tb - m * (ta + tb), m * (A - m) * (B - m),
          (A + B) * (A - m) * (B - m), m * m * (A + B - 2 * m), c * p31 - (B * sa + A * sb) / 3,
          p21 * p21 + p12 * p12 - (B * sa + A * sb - A * B * (A + B) / 4) / 3, p21 * p12 - ta * tb / 3,
          (A + B) * (p22 - A * B / 4)]
    s = 1 + c / 4
    H = [a[i] + b[i] + a[i] * b[i] - c / 4 for i in range(4)]
    N = sum(h * h for h in H)
    assert N == A + B + 2 * c + 2 * ell + p22 - c * c / 4   # N = A + B + 2c + 2l + p22 - c^2/4
    return cs, N, s, c


def main(trials=4, k=2, seed=20261017):
    rng = random.Random(seed)
    worst = 0
    for t in range(trials):
        L = law(k, rng)
        m = sum(w * sum(x * x for x in v) for v, w in L)
        T = sum(w * sum(x ** 3 for x in v) for v, w in L)
        E = [F(0)] * 33
        ENs = F(0); R = F(0); Ec2 = F(0); Ec3 = F(0)
        for (a, wa), (b, wb) in itertools.product(L, L):
            w = wa * wb
            cs, N, s, c = feats(a, b, m)
            for j in range(2, 33):
                E[j] += w * cs[j]
            ENs += w * N / s; R += w * N * c * c / s; Ec2 += w * c * c; Ec3 += w * c ** 3
        nonzero = [j for j in range(2, 33) if E[j] != 0]
        drift = ENs / 16 - (m / 8 - m * T / 48 - T * T / 384 + R / 256)
        print("law", t, "support", len(L), "m", float(m), "T", float(T), "nonzero features:", nonzero,
              "drift residual:", drift, "Ec2-m^2/3:", Ec2 - m * m / 3, "Ec3-2T^2/3:", Ec3 - 2 * T * T / 3, flush=True)
        assert not nonzero and drift == 0 and Ec2 == m * m / 3 and Ec3 == 2 * T * T / 3
    # g = 16/3 - alpha (2 sqrt2 - 2) >= 863/375000000, using sqrt2 < 70711/50000
    alpha = F(32189, 5000); s2 = F(70711, 50000)
    assert s2 * s2 > 2
    g_lo = F(16, 3) - alpha * (2 * s2 - 2)
    print("g lower bound from sqrt2 < 70711/50000:", g_lo, float(g_lo), ">= 863/375000000:", g_lo >= F(863, 375000000))
    # implication: K>=0 pointwise => M0 >= beta m^3 (T >= 0) => x - x_out = M0/256 >= m^3/256e6 = 512 x^3/256e6 = x^3/500000
    assert F(512, 256 * 10 ** 6) == F(1, 500000)
    print("ALL PASS")


if __name__ == "__main__":
    main()
