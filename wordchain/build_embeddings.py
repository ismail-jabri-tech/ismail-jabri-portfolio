#!/usr/bin/env python3
"""
build_embeddings.py — offline pipeline for the Widest Path word game.

Downloads GloVe vectors, filters down to a playable vocabulary, fixes the
anisotropy that makes raw GloVe cosines useless as a score, quantizes to
int8, and emits three static artifacts the browser can load directly:

    data/vocab.json     ["anchor", "ballast", ...]              (~150 KB)
    data/vectors.bin    int8, row-major, shape (N, DIM)         (~1.2 MB)
    data/bias.bin       float32, the CSLS hubness term           (~48 KB)
    data/meta.json      { dim, count, scale, target, lo, hi }

Because every vector is L2-normalized before quantization, cosine similarity
in the browser collapses to a plain dot product. No math library client-side.

Dependencies: numpy. That's the whole list.

    pip install numpy
    python build_embeddings.py --out data --vocab-size 12000

nltk is used if present to filter out proper nouns, but is optional.

Run this once, commit the artifacts, never think about it again.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np

# The gensim-data release assets are just gzipped word2vec-format text files.
# We fetch them straight from GitHub rather than installing gensim, which drags
# in scipy and starts a version fight with whatever numpy you already have.
MODELS = {
    "glove-wiki-gigaword-50":  ("glove-wiki-gigaword-50",   66_000_000),
    "glove-wiki-gigaword-100": ("glove-wiki-gigaword-100", 134_300_434),
    "glove-wiki-gigaword-200": ("glove-wiki-gigaword-200", 265_000_000),
}
BASE_URL = ("https://github.com/RaRe-Technologies/gensim-data/"
            "releases/download/{name}/{name}.gz")

# Words that are technically common but make for miserable gameplay: function
# words, pronouns, auxiliaries, filler. GloVe ranks these highest by frequency,
# so they would otherwise dominate the vocabulary.
STOPLIST = set("""
a about above after again against all am an and any are aren as at be because
been before being below between both but by can cannot could couldn did didn do
does doesn doing don down during each few for from further had hadn has hasn
have haven having he her here hers herself him himself his how i if in into is
isn it its itself just let ll me more most mustn my myself no nor not now of off
on once only or other ought our ours ourselves out over own re s same shan she
should shouldn so some such t than that the their theirs them themselves then
there these they this those through to too under until up ve very was wasn we
were weren what when where which while who whom why will with won would wouldn
you your yours yourself yourselves also however therefore thus hence moreover
said says say get got gets getting one two three four five six seven eight nine
ten first second third new old like many much even still back way well going
made make makes take takes come comes go goes see sees know knows think thinks
john corp inc ltd co reuters http www
""".split())


def download(model: str, cache_dir: Path) -> Path:
    """Fetch the .gz once and cache it next to the script."""
    name, expected = MODELS[model]
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{name}.gz"

    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[1/8] Using cached {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest

    url = BASE_URL.format(name=name)
    print(f"[1/8] Downloading {name} (~{expected / 1e6:.0f} MB)...")

    tmp = dest.with_suffix(".gz.part")
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length") or expected)
            done = 0
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                bars = int(36 * done / total)
                sys.stdout.write(
                    f"\r      [{'=' * bars}{' ' * (36 - bars)}] "
                    f"{done / 1e6:5.0f} / {total / 1e6:.0f} MB")
                sys.stdout.flush()
        print()
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        sys.exit(f"\nDownload failed: {exc}\n"
                 f"You can grab it manually from:\n  {url}\n"
                 f"and save it to: {dest}")

    shutil.move(tmp, dest)
    return dest


def wordnet_lexicon() -> set[str] | None:
    """
    Words WordNet recognizes, restricted to lemmas that are natively lowercase.

    This is the highest-leverage filter in the pipeline. GloVe's frequency-ranked
    vocabulary is full of surnames, place names, ticker symbols, and web detritus
    ("http", "reuters", "ltd") that is unguessable and unfun.

    The lowercase restriction is the important half. Plain WordNet membership
    still admits "london", "stuttgart", "mcpherson" and friends, because WordNet
    catalogues proper nouns too — but it stores them capitalized. Keeping only
    lemmas that are already lowercase drops ~40k entries and takes the proper
    nouns with them, while every common noun survives untouched.

    Optional: if nltk isn't available the game still works, it just has a
    junkier word list.
    """
    print("[2/8] Building WordNet lexicon...")
    try:
        import nltk
        try:
            from nltk.corpus import wordnet as wn
            wn.synsets("test")
        except LookupError:
            print("      downloading wordnet corpus...")
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            from nltk.corpus import wordnet as wn
            wn.synsets("test")
        # .islower() is the proper-noun filter; see docstring.
        return {l.name() for s in wn.all_synsets() for l in s.lemmas()
                if l.name().islower()}
    except Exception as exc:
        print(f"      SKIPPED — nltk unavailable ({type(exc).__name__}: {exc})")
        print("      Vocabulary will contain proper nouns. `pip install nltk` to fix.")
        return None


def stream_vectors(path: Path, lexicon: set[str] | None, size: int):
    """
    Walk the file in frequency order, keeping the first `size` playable words.

    The file is sorted most-common-first, so we can stop as soon as we have
    enough and never load the remaining ~390k vectors. Peak memory stays in
    the tens of MB rather than gigabytes.
    """
    print(f"[3/8] Streaming and curating (target {size:,} words)...")
    vocab: list[str] = []
    rows: list[np.ndarray] = []

    with gzip.open(path, "rt", encoding="utf-8") as fh:
        first = fh.readline()
        # word2vec format opens with "<count> <dim>"; plain GloVe dives straight
        # into data. Detect which, and don't lose the line if there's no header.
        parts = first.split()
        is_header = len(parts) == 2 and all(p.isdigit() for p in parts)
        lines = fh if is_header else itertools.chain([first], fh)

        for line in lines:
            if len(vocab) >= size:
                break
            word, _, rest = line.partition(" ")
            if not (3 <= len(word) <= 12):
                continue
            if not word.isalpha() or not word.isascii():
                continue
            if word in STOPLIST:
                continue
            if lexicon is not None and word not in lexicon:
                continue
            vocab.append(word)
            rows.append(np.array(rest.split(), dtype=np.float32))

    if len(vocab) < size:
        print(f"      note: only {len(vocab):,} words survived filtering")
    return vocab, np.stack(rows)


def center(matrix: np.ndarray) -> np.ndarray:
    """
    Mean-center, then L2-normalize.

    Centering matters more than it looks. Raw GloVe space is strongly
    anisotropic: every vector shares a large common component, so two completely
    unrelated words still score ~75% cosine. That makes the percentage shown to
    the player meaningless — everything reads "somewhat related." Subtracting the
    mean vector removes the shared component and recenters random pairs near 0%,
    which is what a player intuitively expects. It's the cheap half of the
    "all-but-the-top" post-processing from Mu & Viswanath (2018).
    """
    print("[4/8] Mean-centering and normalizing...")
    matrix = matrix.astype(np.float32) - matrix.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def quantize(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """
    int8 quantization: 4x smaller than float32 for an error well under the
    game's display precision. Scaling by the true max beats percentile clipping
    here — clipping the outlier components costs more than the finer step buys.
    """
    print("[5/8] Quantizing to int8...")
    scale = float(np.abs(matrix).max()) / 127.0
    q = np.clip(np.round(matrix / scale), -127, 127).astype(np.int8)

    restored = q.astype(np.float32) * scale
    restored /= np.linalg.norm(restored, axis=1, keepdims=True)
    drift = np.abs((restored * matrix).sum(axis=1) - 1.0).max()
    print(f"      max drift from unit norm: {drift:.6f}")
    return q, scale


def hubness(matrix: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Per-word hubness term r[x]: the mean cosine to x's k nearest neighbours.

    This exists because mean-centering fixes the *offset* of the similarity
    scale but not its *shape*. High-dimensional embeddings have hubs — words
    that sit close to everything — and they are exactly the high-frequency
    words GloVe ranks first: "year" (r=0.75), "time" (0.69), "good" (0.72).

    Left uncorrected, the optimal strategy in this game is to spam generic
    filler, because "time" is 49% similar to any two things you like. That is a
    boring game. CSLS (Conneau et al. 2017) subtracts each word's hubness from
    every score it participates in:

        csls(a, b) = 2 * cos(a, b) - r[a] - r[b]

    A hub pays for its own popularity, and "pacific / island / sea" beats
    "time / good / year" as a way to connect trading to boat.
    """
    print("[6/8] Measuring hubness for CSLS...")
    n = len(matrix)
    r = np.zeros(n, np.float32)
    for i in range(0, n, 1000):
        S = matrix[i:i + 1000] @ matrix.T
        for j in range(S.shape[0]):
            S[j, i + j] = -9.0                 # never a neighbour of itself
        r[i:i + 1000] = np.sort(S, axis=1)[:, -k:].mean(axis=1)
    return r


def calibrate(matrix: np.ndarray, r: np.ndarray) -> tuple[float, float]:
    """
    Map raw CSLS onto a 0-100% scale a player can read.

    Anchored on two empirical landmarks: the median random pair becomes 0%,
    and the 90th percentile of nearest-neighbour pairs becomes 100%. Scores
    can go negative — "piano / hurricane" lands near -30% — which is honest.
    They really are further apart than chance.
    """
    print("[7/8] Calibrating the display scale...")
    rng = np.random.default_rng(3)
    n = len(matrix)

    pairs = rng.integers(0, n, (20000, 2))
    rand = (2 * np.einsum("ij,ij->i", matrix[pairs[:, 0]], matrix[pairs[:, 1]])
            - r[pairs[:, 0]] - r[pairs[:, 1]])

    near = []
    for i in rng.integers(0, n, 600):
        s = matrix[i] @ matrix.T
        s[i] = -9.0
        j = int(s.argmax())
        near.append(2 * float(s[j]) - r[i] - r[j])

    lo = float(np.percentile(rand, 50))
    hi = float(np.percentile(near, 90))
    print(f"      random median {lo:.3f} -> 0%   strong pair p90 {hi:.3f} -> 100%")
    return lo, hi


def report(matrix: np.ndarray, vocab: list[str],
           r: np.ndarray, lo: float, hi: float) -> None:
    """Sanity check: do the numbers pass the smell test?"""
    idx = {w: i for i, w in enumerate(vocab)}

    def score(a, b):
        c = 2 * float(matrix[a] @ matrix[b]) - r[a] - r[b]
        return (c - lo) / (hi - lo)

    probes = [("king", "queen"), ("infection", "disease"), ("river", "bank"),
              ("hard", "time"), ("piano", "hurricane"), ("dog", "asphalt")]

    print("\n      scored probes")
    for a, b in probes:
        if a in idx and b in idx:
            print(f"        {a:>10} ~ {b:<10} {score(idx[a], idx[b]) * 100:6.1f}%")

    rng = np.random.default_rng(0)
    pairs = rng.integers(0, len(vocab), size=(4000, 2))
    base = np.array([score(int(x), int(y)) for x, y in pairs])
    print(f"        random pair median: {np.median(base) * 100:5.1f}% "
          f"(should sit near 0)\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data")
    p.add_argument("--vocab-size", type=int, default=12000)
    p.add_argument("--model", default="glove-wiki-gigaword-100", choices=MODELS)
    p.add_argument("--cache", default=".cache", help="where to keep the GloVe download")
    p.add_argument("--target", type=float, default=0.45,
                   help="bottleneck similarity needed to win")
    args = p.parse_args()

    src = download(args.model, Path(args.cache))
    lexicon = wordnet_lexicon()
    vocab, raw = stream_vectors(src, lexicon, args.vocab_size)
    matrix = center(raw)
    q, scale = quantize(matrix)
    r = hubness(matrix)
    lo, hi = calibrate(matrix, r)
    report(matrix, vocab, r, lo, hi)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[8/8] Writing to {out.resolve()}...")
    (out / "vocab.json").write_text(json.dumps(vocab, separators=(",", ":")))
    (out / "vectors.bin").write_bytes(q.tobytes())
    (out / "bias.bin").write_bytes(r.astype(np.float32).tobytes())
    (out / "meta.json").write_text(json.dumps({
        "dim": int(matrix.shape[1]),
        "count": len(vocab),
        "scale": scale,
        "target": args.target,
        "lo": lo,
        "hi": hi,
        "source": args.model,
    }, indent=2))

    total = sum((out / f).stat().st_size
                for f in ("vocab.json", "vectors.bin", "bias.bin", "meta.json"))
    print(f"\nDone. {len(vocab):,} words, {total / 1e6:.2f} MB total.")


if __name__ == "__main__":
    main()
