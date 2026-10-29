# Lantern

Lantern is an offline-first retrieval-augmented generation CLI. It chunks your
markdown notes into overlapping windows, embeds each chunk with a hashing-trick
vectorizer, and answers questions with citations — all without network access
or API keys.

## Ingest pipeline

Running `lantern ingest docs` walks the directory, reads every `.md` and `.txt`
file, and splits it into chunks. The chunker is heading-aware: it starts a new
section at every ATX heading and keeps the heading path on each chunk. Long
sections are cut into fixed-size windows with overlap so context is never lost
at a boundary.

Every chunk records its source file plus a 1-based inclusive line range, so a
citation can always be traced back to the original document.

## Embeddings

The embedder tokenizes text into lowercase alphanumeric tokens and builds token
n-grams from one to three tokens. Each n-gram is hashed with FNV-1a into a fixed
256-dimensional vector using signed projection, and the result is L2-normalized.
The same text always produces the same vector, which makes the whole pipeline
deterministic and fully offline.

## Retrieval

The vector index stores chunk embeddings and scores a query by brute-force
cosine similarity. The index serializes to a single `index.json` file that can
be committed or shipped alongside your docs. `lantern query "..." -k 5` prints
the top-k chunks with their score and `file:startLine-endLine` citation.

## Design goals

Lantern favors transparency over magic: every stage is small, readable, and
testable. The embedder is a hashing trick rather than a neural model, so it
needs no model weights, no downloads, and no GPU. Retrieval is brute force,
which is plenty fast for a personal knowledge base and trivially correct.

## Roadmap

Day two adds a chat REPL with [n] citations and a pluggable LLM provider that
defaults to a deterministic extractive mock, plus a JSONL eval harness that
reports recall@3 and grounded-answer scores as a markdown report.
