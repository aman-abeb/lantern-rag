# Lantern

<!-- builder:badges:start -->
[![CI](https://github.com/aman-abeb/lantern-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/aman-abeb/lantern-rag/actions/workflows/ci.yml) ![TypeScript](https://img.shields.io/badge/typescript-strict-3178C6?logo=typescript&logoColor=white) ![Node.js](https://img.shields.io/badge/node.js-20%2B-5FA04E?logo=nodedotjs&logoColor=white) ![License: MIT](https://img.shields.io/badge/license-MIT-green)
<!-- builder:badges:end -->


Offline-first RAG CLI: chunk your docs, retrieve with a from-scratch vector index, and get answers with citations — no network calls, no API keys.

## Features

- **Heading-aware chunker** — splits Markdown on ATX headings, windows long sections with fixed-size + overlap, and attaches `file` + 1-based line ranges to every chunk
- **Hashing-trick embedder** — token n-grams (1–3) hashed with FNV-1a into a fixed 256-dim, L2-normalized vector; fully deterministic, zero downloads
- **Brute-force vector index** — cosine similarity over stored embeddings, serialized to a single `index.json`
- **CLI** — `lantern ingest <dir>` and `lantern query <text> -k 5`, with `[n]` results mapped to `file:startLine-endLine`
- **Offline by default** — `npm test` makes zero network calls and needs zero API keys

## Quickstart

```bash
npm install
npm run build
npm start -- ingest test/fixtures
npm start -- query 'how does the hashing embedder work?' -k 3
```

During development you can skip the build step with `npm run dev -- ingest test/fixtures` (uses `tsx`).

## Example

Real transcript against the bundled fixture (after `npm run build`):

```text
$ node dist/cli.js ingest test/fixtures
Ingested 1 file(s) → 6 chunk(s) (dim=256)
Index written to index.json

$ node dist/cli.js query 'how does the hashing embedder work?' -k 3
[1] 0.421  test/fixtures/sample.md:19-26
    ## Embeddings The embedder tokenizes text into lowercase alphanumeric tokens and builds token n-grams from one to three tokens. Each n-gram is hashed with FNV-1a into a fixed 256-dimensional vector using signed proj…
[2] 0.268  test/fixtures/sample.md:8-18
    ## Ingest pipeline Running `lantern ingest docs` walks the directory, reads every `.md` and `.txt` file, and splits it into chunks. The chunker is heading-aware: it starts a new section at every ATX heading and keeps the…
[3] 0.154  test/fixtures/sample.md:1-7
    # Lantern Lantern is an offline-first retrieval-augmented generation CLI. It chunks your markdown notes into overlapping windows, embeds each chunk with a hashing-trick vectorizer, and answers questions with cit…
```

Each `[n]` maps to `file:startLine-endLine`, so every citation resolves to an exact source location.

## How it works

1. **Chunking** (`src/chunker.ts`) — the document is split into sections at every `#`–`######` heading (keeping the heading path). Sections longer than `maxChars` (default 900) are cut into windows that repeat ~`overlapChars` (default 150) of trailing text. Every chunk stores `file`, `startLine`, `endLine`, and a stable content-hash id.
2. **Embedding** (`src/embedder.ts`) — text is lowercased, tokenized on alphanumeric runs, and expanded into token n-grams (sizes 1–3). Each n-gram is hashed with FNV-1a; the hash picks a dimension and a sign (signed projection), then the vector is L2-normalized. Same text → same vector, always.
3. **Index** (`src/vector-index.ts`) — chunks plus embeddings live in a `VectorIndex` that ranks queries by brute-force cosine similarity and serializes to `index.json` (`VectorIndex.save` / `VectorIndex.load`).
4. **CLI** (`src/cli.ts`) — `ingest` walks a directory for `.md`/`.txt` files, chunks and embeds them, and writes the index; `query` embeds the question and prints the top-k chunks with score and `file:line` citation.

## Running tests

```bash
npm test        # vitest run — chunker, embedder, and index recall tests (fully offline)
npm run build   # tsc type-check + emit to dist/
```

The tests cover chunker boundaries/overlap/line metadata, embedder determinism and unit-norm, and index recall@3 on the bundled fixture corpus — all with zero network access.

<!-- builder:roadmap:start -->
## Status

**In active development:** 1 of 2 milestones done.

## Roadmap

- [x] **Ingest & retrieve pipeline**: Markdown/text chunker (heading-aware, fixed-size with overlap, source line metadata), hashing-trick embedder (token n-grams hashed into a fixed-dim L2-normalized vector, no network), cosine-similarity brute-force vector index serialized to index.json, plus `lantern ingest <dir>` and `lantern query <text> -k 5` CLI commands. Unit tests for chunker boundaries, embedder determinism/normalization, and index recall on a fixture corpus; runnable end-to-end on day 1.
- [ ] **Chat with citations + offline eval harness**: LLMProvider interface with a deterministic extractive MockProvider (default, composes answers from retrieved chunks) and an optional real HTTP provider enabled only via env vars; `lantern chat` REPL with conversation history and [n] citation formatting that resolves to file:line ranges; JSONL eval harness over a bundled QA set computing recall@k and grounded-answer metrics with a mock judge, emitting a markdown report. Tests for citation resolution, provider fallback, and eval metrics.
<!-- builder:roadmap:end -->
