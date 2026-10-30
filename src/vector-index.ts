import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { EMBEDDING_DIM, cosine } from './embedder.js';
import type { Chunk, IndexedChunk, LanternIndex, SearchResult } from './types.js';

/** Version of the on-disk index schema. */
export const INDEX_VERSION = 1;
/** Identifier of the embedding recipe recorded in the index. */
export const INDEX_MODEL = 'hashing-ngram-fnv1a';

/**
 * Brute-force vector index: stores chunks with their embeddings and ranks them
 * by cosine similarity to a query embedding. Serializes to a single JSON file.
 */
export class VectorIndex {
  /** Embedding dimensionality shared by every chunk in this index. */
  readonly dim: number;
  private readonly chunks: IndexedChunk[] = [];

  constructor(dim: number = EMBEDDING_DIM) {
    if (!Number.isInteger(dim) || dim < 1) throw new Error(`Invalid embedding dimension: ${dim}`);
    this.dim = dim;
  }

  /** Number of chunks in the index. */
  get size(): number {
    return this.chunks.length;
  }

  /** Add a chunk with its pre-computed embedding (must match `dim`). */
  add(chunk: Chunk, embedding: number[]): void {
    if (embedding.length !== this.dim) {
      throw new Error(`Embedding dimension mismatch: expected ${this.dim}, got ${embedding.length}`);
    }
    this.chunks.push({ ...chunk, embedding: [...embedding] });
  }

  /**
   * Return the top-`k` chunks for a query embedding, sorted by descending
   * cosine similarity (ties broken by chunk id for determinism).
   */
  search(query: number[], k: number): SearchResult[] {
    if (query.length !== this.dim) {
      throw new Error(`Query dimension mismatch: expected ${this.dim}, got ${query.length}`);
    }
    const scored = this.chunks.map((chunk) => ({ chunk, score: cosine(query, chunk.embedding) }));
    scored.sort((a, b) => b.score - a.score || a.chunk.id.localeCompare(b.chunk.id));
    return scored.slice(0, Math.max(0, k));
  }

  /** Look up a chunk by its stable id; undefined when absent. */
  findById(id: string): IndexedChunk | undefined {
    return this.chunks.find((chunk) => chunk.id === id);
  }

  /** Look up the chunk that starts at `startLine` of `file`; undefined when absent. */
  findByFileLine(file: string, startLine: number): IndexedChunk | undefined {
    return this.chunks.find((chunk) => chunk.file === file && chunk.startLine === startLine);
  }

  /** Serialize to the on-disk {@link LanternIndex} schema. */
  toJSON(): LanternIndex {
    return {
      version: INDEX_VERSION,
      dim: this.dim,
      model: INDEX_MODEL,
      createdAt: new Date().toISOString(),
      chunks: this.chunks,
    };
  }

  /** Rebuild an index from a parsed {@link LanternIndex} object. */
  static fromJSON(data: LanternIndex): VectorIndex {
    if (data.version !== INDEX_VERSION) throw new Error(`Unsupported index version: ${String(data.version)}`);
    if (!Array.isArray(data.chunks)) throw new Error("Invalid index file: 'chunks' must be an array");
    const index = new VectorIndex(data.dim);
    for (const chunk of data.chunks) index.add(chunk, chunk.embedding);
    return index;
  }

  /** Write the index to `path` as pretty-printed JSON, creating parent directories. */
  save(path: string): void {
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, `${JSON.stringify(this.toJSON(), null, 2)}\n`, 'utf8');
  }

  /** Load an index previously written with {@link VectorIndex.save}. */
  static load(path: string): VectorIndex {
    const raw = readFileSync(path, 'utf8');
    return VectorIndex.fromJSON(JSON.parse(raw) as LanternIndex);
  }
}
