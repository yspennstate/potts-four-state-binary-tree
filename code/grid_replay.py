"""Recomputes the grid bound of Proposition 3.1 and compares it with data/grid_certificate_N36.json.

Four-state Potts, binary tree, lambda = A/B = 707107/1000000, grid N = 36, masses in units of 2^-48, depth 500.
State = S4-orbit of a grid vector v/N (sorted key). One step: for each unordered orbit pair (a, b), fix the first
orbit's sorted representative v, average over every distinct orientation w of the second, form the parent
posterior Z/C with Z_i = h_i(v) h_i(w), h_i = (B-A)N + 4A v_i, split it onto the grid by the symmetric rule of
section 1, and weight by C/(4D), D = (BN)^2. Grouped numerators over common = 288 D. New orbit masses are
floor(sum / (scale*common)); the missing mass goes to full revelation (0,0,0,N).

Checks while building: every split has exact barycenter and nonnegative weights summing to one; every pair's
grouped numerators sum to common. Output: per-depth x_upper fractions, compared to the certificate.
"""
import itertools, json, sys, time
from array import array
from fractions import Fraction as F
from pathlib import Path

A, B, N, BITS, DEPTH = 707107, 1000000, 36, 48, 500
HERE = Path(__file__).resolve().parent
CERT = HERE / "data" / "grid_certificate_N36.json"
LOG = HERE / "results" / "grid_replay.log"
D = (B * N) ** 2
COMMON = 288 * D
SCALE = 1 << BITS


def say(*a):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a) + "\n")


keys = sorted(tuple(sorted(c)) for c in
              {tuple(sorted((a, b - a, c - b, N - c))) for a in range(N + 1) for b in range(a, N + 1) for c in range(b, N + 1)})
index = {k: i for i, k in enumerate(keys)}
orients = [sorted(set(itertools.permutations(k))) for k in keys]
assert all(24 % len(o) == 0 for o in orients)


def split(Z):
    """Split numerators (common denominator 3C) of the posterior Z/C onto grid vectors, by the section-1 rule."""
    C = sum(Z)
    f = [N * z // C for z in Z]
    r = [N * z - fi * C for z, fi in zip(Z, f)]   # rnum_i = C * r_i
    R, rem = divmod(sum(r), C)
    assert rem == 0 and 0 <= R <= 3
    out = []
    if R == 0:
        out.append((3 * C, tuple(f)))
    elif R == 1:
        for i in range(4):
            if r[i]:
                v = list(f); v[i] += 1; out.append((3 * r[i], tuple(v)))
    elif R == 3:
        for i in range(4):
            if C - r[i]:
                v = [x + 1 for x in f]; v[i] -= 1; out.append((3 * (C - r[i]), tuple(v)))
    else:
        h = min(min(r), C - max(r))
        for i, j in itertools.combinations(range(4), 2):
            wnum = 3 * max(r[i] + r[j] - C, 0) + h
            if wnum:
                v = list(f); v[i] += 1; v[j] += 1; out.append((wnum, tuple(v)))
    assert all(w > 0 for w, _ in out)
    assert sum(w for w, _ in out) == 3 * C
    for i in range(4):  # exact barycenter: sum w * v_i = 3 C * N * (Z_i / C)
        assert sum(w * v[i] for w, v in out) == 3 * N * Z[i]
    return out


def build():
    t0 = time.time()
    pairs = []  # (a, b, factor, dest array, coef array)
    hv = [[(B - A) * N + 4 * A * x for x in k] for k in keys]
    for a in range(len(keys)):
        v = hv[a]
        for b in range(a, len(keys)):
            acc = {}
            mult = 24 // len(orients[b])
            for w in orients[b]:
                Z = [v[i] * ((B - A) * N + 4 * A * w[i]) for i in range(4)]
                for wnum, out in split(Z):
                    o = index[tuple(sorted(out))]
                    acc[o] = acc.get(o, 0) + wnum * mult
            assert sum(acc.values()) == COMMON, (a, b)
            ds = array("H", acc.keys()); cs = array("q", acc.values())
            pairs.append((a, b, 1 if a == b else 2, ds, cs))
        if a % 50 == 0:
            say("build", a, "/", len(keys), round(time.time() - t0), "s")
    nnz = sum(len(p[3]) for p in pairs)
    say("built", len(pairs), "pairs, nnz", nnz, round(time.time() - t0), "s")
    return pairs


def moment(m):
    return F(sum(mi * (4 * sum(x * x for x in k) - N * N) for mi, k in zip(m, keys)), SCALE * 4 * N * N)



def main():
    LOG.write_text("", encoding="utf-8")
    cert = json.loads(CERT.read_text(encoding="utf-8"))
    assert cert["keys"] == [list(k) for k in keys], "orbit keys differ from the certificate's"
    pairs = build()
    full = index[(0, 0, 0, N)]
    m = [0] * len(keys); m[full] = SCALE
    rows, agree = [], 0
    t0 = time.time()
    for depth in range(1, DEPTH + 1):
        num = [0] * len(keys)
        for a, b, fac, ds, cs in pairs:
            p = m[a] * m[b]
            if not p:
                continue
            p *= fac
            for o, c in zip(ds, cs):
                num[o] += c * p
        assert sum(num) == SCALE * SCALE * COMMON
        new = [x // (SCALE * COMMON) for x in num]
        miss = SCALE - sum(new)
        new[full] += miss
        m = new
        x = moment(m)
        theirs = F(cert["rows"][depth - 1]["x_upper"])
        same = x == theirs and F(miss, SCALE) == F(cert["rows"][depth - 1]["rounding_revelation_mass"])
        agree += same
        rows.append(dict(depth=depth, x=str(x), same=same))
        if depth % 10 == 0 or not same or depth <= 3:
            say("depth", depth, float(x), "same as certificate:", same, round(time.time() - t0), "s")
        if depth % 25 == 0:
            (HERE / "results" / "grid_replay.json").write_text(json.dumps(dict(rows=rows, agree=agree), indent=0), encoding="utf-8")
    final = moment(m)
    res = dict(rows=rows, agree=agree, final=str(final), final_float=float(final),
               masses_equal=m == cert["masses"], claim=str(F(1375940705703331, 30399297484750848)),
               final_equals_claim=final == F(1375940705703331, 30399297484750848), below_23_500=final < F(23, 500))
    (HERE / "results" / "grid_replay.json").write_text(json.dumps(res, indent=0), encoding="utf-8")
    say("DONE agree", agree, "/", DEPTH, "final", float(final), "equals claim", res["final_equals_claim"],
        "masses equal", res["masses_equal"])


if __name__ == "__main__":
    main()
