# Widest Path

A word-chain game for the portfolio. Two unrelated words sit at either end of a
board; you add words until meaning flows between them. Your score is the weakest
link on your best route.

Inspired by [Linxicon](https://linxicon.com) by Trainwreck Labs — built from
scratch, with its own name, wording, and vocabulary. Go play theirs too.

## Why this is actually a data engineering project

- **The score is a graph problem.** The board is a complete weighted graph where
  edges are similarities. Your best route is the *widest path* — Dijkstra's
  algorithm with min-plus swapped for max-min. The bottleneck edge is the score,
  and a link is only *drawn* once it beats that bottleneck, which is why the two
  halves stay visibly apart until you've genuinely bridged them.
- **Mean-centering fixes the offset.** Raw GloVe is anisotropic: every vector
  shares a large common component, so two unrelated words still score ~75% cosine.
  Subtracting the mean vector drops random pairs to ~0%.
- **CSLS fixes the shape.** Centering isn't enough. High-dimensional embeddings
  have *hubs* — words near everything — and they're exactly the words GloVe ranks
  first: `year` (r=0.75), `good` (0.72), `time` (0.69). Uncorrected, the winning
  strategy is spamming filler, because `time` is ~49% similar to any two things
  you like. CSLS charges each word for its own popularity, and the game turns
  into `pacific → island → sea` instead of `time → good → year`.
- **The whole thing is static.** The pipeline emits 1.37 MB of int8 + float32;
  the browser dequantizes once and does dot products. No server, no libraries —
  the force-directed layout is hand-rolled on a `<canvas>`.

## Setup

```bash
pip install numpy nltk
python build_embeddings.py --out data --vocab-size 12000
```

No gensim — it drags in scipy and starts a version fight with whatever numpy you
already have. The GloVe vectors are fetched straight from the gensim-data GitHub
release (134 MB, cached in `.cache/`, gzipped word2vec text). nltk is optional but
strongly recommended; without it the word list keeps proper nouns.

It writes:

```
data/vocab.json     12k curated words          122 KB
data/vectors.bin    int8, 12000 x 100          1.2 MB
data/bias.bin       float32, CSLS hubness       48 KB
data/meta.json      dim, scale, target, lo/hi
```

Commit those four files. Then serve — it needs HTTP, not `file://`, because of
`fetch`:

```bash
python -m http.server 8000
```

## Tuning

| Flag | Default | Effect |
|---|---|---|
| `--vocab-size` | `12000` | Bigger accepts more words but bloats the payload |
| `--target` | `0.45` | Bottleneck needed to win. See calibration below |
| `--model` | `glove-wiki-gigaword-100` | `-200` doubles quality and payload |
| `--cache` | `.cache` | Where the GloVe download lives. Add to `.gitignore` |

### Calibrating `--target`

Don't calibrate against a perfect solver searching all 12k words — no human plays
that way, and it will hand you an unwinnable target. Calibrate against a greedy
bot restricted to the ~1200 commonest words: exhaustive over a pool a person
actually has at hand, so it's a *ceiling* on human play.

| Words played | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Median gap | 46% | 49% | 54% | 58% | 60% | 60% | 62% | 62% |
| Worst round | 33% | 41% | 43% | 44% | 49% | 49% | 49% | 49% |

The worst-round row is the one that matters — it's what makes a target unwinnable.

| Target | Bot words needed | Rounds winnable at all |
|---|---|---|
| 45% | 3 | 100% |
| 50% | 4 | 90% |
| 55% | 6 | 90% |
| 60% | >8 | 70% |

`0.45` is the default: the bot needs three words, a human lands around five to
eight, and every round is winnable. `0.50` is a real challenge but silently
bricks one round in ten. Anything at or above `0.60` ships a broken game.

## Reading the graph

- **Blue** is the component around your first word, **purple** the second.
- A link is drawn only when it beats the current gap, so the two colours can
  never touch. The **red dashed line is the gap** — the hole you're closing.
- Grey pills are words you played that the best route ignores.
- Drag anything.

## Accuracy note

int8 quantization costs ~0.13 percentage points on average (~0.6pp worst case),
invisible at the one decimal shown. The CSLS bias is kept as float32 — it's only
48 KB, and it's subtracted from every score, so it isn't worth approximating.

## Portfolio integration

**Button** — next to the hero CTAs in `index.html`:

```html
<a href="wordchain/" class="btn-o btn-game">◈ Play Widest Path</a>
```

```css
.btn-game {
  border-color: rgba(0,212,255,.22);
  background: var(--ring2);
  color: var(--ice);
}
.btn-game:hover {
  border-color: var(--ice);
  box-shadow: 0 0 20px var(--ring2);
  transform: translateY(-2px);
}
```

**Project card** — drop into the `.proj-grid` in your `#projects` section:

```html
<div class="pj fi">
  <div class="pj-hdr"><div class="pj-icon">🔗</div></div>
  <h3 class="pj-title">Widest Path — Word Chain Game</h3>
  <p class="pj-desc">
    A browser word game scored by the bottleneck edge of a semantic graph.
    Offline pipeline mean-centers and quantizes GloVe vectors to a 1.2 MB int8
    blob; the client solves a max-min Dijkstra on every turn. Fully static.
  </p>
  <div class="pj-tags">
    <span class="pt">Python</span><span class="pt">NumPy</span>
    <span class="pt">Embeddings</span><span class="pt">Quantization</span>
    <span class="pt">Graph Algorithms</span>
  </div>
</div>
```

Repo layout this assumes:

```
ismail-jabri-portfolio/
├── index.html
├── assets/
└── wordchain/
    ├── index.html
    ├── build_embeddings.py
    └── data/
```

The back-link in the game points at `/ismail-jabri-portfolio/`. Change it if you
ever move to a custom domain.
