# Lantern

<!-- builder:badges:start -->
[![CI](https://github.com/aman-abeb/lantern-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/aman-abeb/lantern-rag/actions/workflows/ci.yml) ![TypeScript](https://img.shields.io/badge/typescript-strict-3178C6?logo=typescript&logoColor=white) ![Node.js](https://img.shields.io/badge/node.js-20%2B-5FA04E?logo=nodedotjs&logoColor=white) ![License: MIT](https://img.shields.io/badge/license-MIT-green)
<!-- builder:badges:end -->


Offline-first RAG CLI: chunk your docs, retrieve with a from-scratch vector index, and chat with citations via a pluggable LLM provider — no network calls, no API keys.

## Features

- **Heading-aware chunker** — splits Markdown on ATX headings, windows long sections with fixed-size + overlap, and attaches `file` + 1-based line ranges to every chunk
- **Hashing-trick embedder** — token n-grams (1–3) hashed with FNV-1a into a fixed 256-dim, L2-normalized vector; fully deterministic, zero downloads
- **Brute-force vector index** — cosine similarity over stored embeddings, serialized to a single `index.json`
- **Chat with citations** — `lantern chat` runs a REPL with conversation history; answers carry `[n]` markers that resolve to `file:startLine-endLine` in a Sources footer
- **Pluggable LLM provider** — deterministic extractive `MockProvider` by default; an HTTP provider activates only when `LANTERN_LLM_URL` (plus optional `LANTERN_LLM_API_KEY` / `LANTERN_LLM_MODEL`) is set
- **Offline eval harness** — `lantern eval` runs a JSONL QA set and prints recall@3 + grounded-answer as a markdown report
- **Offline by default** — `npm test` makes zero network calls and needs zero API keys

## Quickstart

```bash
npm install
npm run build
npm start -- ingest test/fixtures
npm start -- query 'how does the hashing embedder work?' -k 3
npm start -- chat -k 3
npm start -- eval -k 3
```

During development you can skip the build step with `npm run dev -- <command>` (uses `tsx`).

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

### Chat

```text
$ node dist/cli.js chat -k 3
Lantern chat — provider: mock-extractive, index: index.json (type /quit or Ctrl-D to exit)
> What line metadata does each chunk record?
Every chunk records its source file plus a 1-based inclusive line range, so a citation can always be traced back to the original document. [1]

Sources:
[1] test/fixtures/sample.md:8-18
> /quit
```

The default mock provider is extractive: it composes its answer from the retrieved chunks and appends `[n]` after each sentence, so every claim stays grounded in a cited `file:line` range. Set `LANTERN_LLM_URL` (plus optional `LANTERN_LLM_API_KEY` / `LANTERN_LLM_MODEL`) to swap in a real HTTP LLM — the `[n]` citation contract stays the same.

### Eval

```text
$ node dist/cli.js eval -k 3
# Lantern eval report

- Provider: mock-extractive
- Questions: 6
- recall@3: 1.000
- grounded-answer: 1.000

| ID | Question | recall@3 | grounded | Top sources |
| --- | --- | ---: | ---: | --- |
| q1 | How does the embedder turn text into a vector? | 1.00 | 1.00 | [1] test/fixtures/sample.md:19-26 |
| q2 | What source metadata does every chunk record? | 1.00 | 1.00 | [1] test/fixtures/sample.md:8-18 |
| q3 | How does the vector index rank chunks for a query? | 1.00 | 1.00 | [1] test/fixtures/sample.md:27-32 |
| q4 | Why does Lantern avoid neural embedding models? | 1.00 | 1.00 | [1] test/fixtures/sample.md:34-39 |
| q5 | What does the day-two roadmap add? | 1.00 | 1.00 | [1] test/fixtures/sample.md:41-45 |
| q6 | What does lantern ingest do with a directory of docs? | 1.00 | 1.00 | [1] test/fixtures/sample.md:8-18 |
```

The QA set lives in `eval/qa.jsonl` — one JSON object per line with `id`, `question`, and `gold` chunk references (`file:startLine` or a chunk id). `recall@3` is the fraction of gold chunks retrieved in the top 3; `grounded-answer` is a deterministic mock judge scoring how many answer sentences are supported by the cited chunks. Pass `--report report.md` to also write the markdown report to a file.

## How it works

1. **Chunking** (`src/chunker.ts`) — the document is split into sections at every `#`–`######` heading (keeping the heading path). Sections longer than `maxChars` (default 900) are cut into windows that repeat ~`overlapChars` (default 150) of trailing text. Every chunk stores `file`, `startLine`, `endLine`, and a stable content-hash id.
2. **Embedding** (`src/embedder.ts`) — text is lowercased, tokenized on alphanumeric runs, and expanded into token n-grams (sizes 1–3). Each n-gram is hashed with FNV-1a; the hash picks a dimension and a sign (signed projection), then the vector is L2-normalized. Same text → same vector, always.
3. **Index** (`src/vector-index.ts`) — chunks plus embeddings live in a `VectorIndex` that ranks queries by brute-force cosine similarity and serializes to `index.json` (`VectorIndex.save` / `VectorIndex.load`).
4. **Providers** (`src/provider.ts`) — the `LLMProvider` interface takes the conversation plus the retrieved context and returns text with `[n]` markers. `MockProvider` (default) is a deterministic extractive composer; `HttpProvider` is only constructed when `LANTERN_LLM_URL` is set and talks to any OpenAI-style chat endpoint. `resolveProvider()` picks between them from env vars only.
5. **Chat** (`src/chat.ts`) — `ChatSession` keeps history, retrieves the top-k chunks per question, calls the provider, and `buildChatAnswer` resolves `[n]` markers to citations (`file:startLine-endLine`). `lantern chat` is a stdin REPL (`/quit` or Ctrl-D to exit).
6. **Eval** (`src/eval.ts`) — `lantern eval` loads a JSONL QA set, runs retrieve + answer per question, computes `recallAtK` against gold chunk ids and a grounded-answer score via the mock judge, and prints a markdown report.
7. **CLI** (`src/cli.ts`) — `ingest`, `query`, `chat`, and `eval` commands.

## Running tests

```bash
npm test        # vitest run — chunker, embedder, index, chat, and eval tests (fully offline)
npm run build   # tsc type-check + emit to dist/
```

The tests cover chunker boundaries/overlap/line metadata, embedder determinism and unit-norm, index recall@3 on the bundled fixture corpus, citation resolution to file:line ranges, provider fallback, and eval metrics — all with zero network access.

<!-- builder:roadmap:start -->
## Status

**In active development:** 1 of 2 milestones done.

## Roadmap

- [x] **Ingest & retrieve pipeline**: Markdown/text chunker (heading-aware, fixed-size with overlap, source line metadata), hashing-trick embedder (token n-grams hashed into a fixed-dim L2-normalized vector, no network), cosine-similarity brute-force vector index serialized to index.json, plus `lantern ingest <dir>` and `lantern query <text> -k 5` CLI commands. Unit tests for chunker boundaries, embedder determinism/normalization, and index recall on a fixture corpus; runnable end-to-end on day 1.
- [ ] **Chat with citations + offline eval harness**: LLMProvider interface with a deterministic extractive MockProvider (default, composes answers from retrieved chunks) and an optional real HTTP provider enabled only via env vars; `lantern chat` REPL with conversation history and [n] citation formatting that resolves to file:line ranges; JSONL eval harness over a bundled QA set computing recall@k and grounded-answer metrics with a mock judge, emitting a markdown report. Tests for citation resolution, provider fallback, and eval metrics.
<!-- builder:roadmap:end -->
