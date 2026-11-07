"""Exact check of Lemma 4.2: fixed-point messages lie in rho(4 Delta - 1) with rho = 677/1000."""
from fractions import Fraction as F
import itertools, random

L = F(707107, 1000000)
assert L * L * 2 > 1  # ell > 1/sqrt(2)
R = F(677, 1000)


def f(r):
    return 2 * L * r * (1 + r) / (1 + 3 * r * r)


def fuse_min(r, lam):
    """Exact minimum over the 16 vertex pairs and 4 colours of lam*(4P_i-1)."""
    verts = [[r * (4 * (1 if j == k else 0) - 1) for j in range(4)] for k in range(4)]
    best = None
    for a, b in itertools.product(verts, verts):
        num = [(1 + a[j]) * (1 + b[j]) for j in range(4)]
        s = sum(num)
        for i in range(4):
            v = lam * (4 * num[i] / s - 1)
            best = v if best is None or v < best else best
    return best


# 1. vertex minimum equals -f_lam(r) exactly, at several r
for r in [F(1, 10), F(1, 2), R, L]:
    assert fuse_min(r, L) == -f(r), r
print("vertex minimum = -f(r) exactly at 4 radii")

# 2. iteration from r0 = ell: three updates land at or below 677/1000, two do not
r = L
seq = [r]
for k in range(3):
    r = f(r)
    seq.append(r)
print("r_k:", [float(x) for x in seq])
assert seq[2] > R and seq[3] <= R

# 3. f is increasing on [0,1] (numerator of f' is (1+3r)(1-r))
assert all(f(F(k + 1, 1000)) >= f(F(k, 1000)) for k in range(1000))

# 4. random interior inputs never go below the vertex bound (float check of the linear-fractional step)
random.seed(17)
worst = 0.0
lam = 2 ** -0.5
for _ in range(200000):
    r = random.random()
    def pt():
        w = [random.random() ** 3 for _ in range(4)]
        s = sum(w)
        return [r * (4 * x / s - 1) for x in w]
    a, b = pt(), pt()
    num = [(1 + a[j]) * (1 + b[j]) for j in range(4)]
    s = sum(num)
    lo = min(lam * (4 * x / s - 1) for x in num)
    bound = -2 * lam * r * (1 + r) / (1 + 3 * r * r)
    worst = min(worst, lo - bound)
print("random interior: min(output - bound) =", worst)
assert worst >= -1e-12

# 5. the output set: coordinates >= -s and zero sum <=> s*(4u-1) with u in the simplex
s = F(3, 5)
c = [F(-3, 5), F(-3, 5), F(1, 5), F(1, 1)]
assert sum(c) == 0 and min(c) >= -s
u = [(x / s + 1) / 4 for x in c]
assert min(u) >= 0 and sum(u) == 1
print("Lemma 4.2: all checks PASS")
