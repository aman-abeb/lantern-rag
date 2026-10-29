/** A chunk of source text together with its provenance metadata. */
export interface Chunk {
  /** Stable content hash (sha1 of file + start line + text), 16 hex chars. */
  id: string;
  /** The chunk text; may span multiple source lines. */
  text: string;
  /** Path of the source file, exactly as passed to the chunker. */
  file: string;
  /** 1-based line number of the first line of the chunk (inclusive). */
  startLine: number;
  /** 1-based line number of the last line of the chunk (inclusive). */
  endLine: number;
  /** Heading titles from the document root down to this chunk's section. */
  headingPath: string[];
}

/** A chunk stored in the vector index together with its embedding vector. */
export interface IndexedChunk extends Chunk {
  /** L2-normalized embedding of `text`, of length `LanternIndex.dim`. */
  embedding: number[];
}

/** A single search hit: the chunk plus its cosine similarity to the query. */
export interface SearchResult {
  chunk: IndexedChunk;
  /** Cosine similarity between the query embedding and the chunk embedding, in [-1, 1]. */
  score: number;
}

/** On-disk schema of `index.json`. */
export interface LanternIndex {
  version: 1;
  /** Embedding dimensionality shared by every chunk in the index. */
  dim: number;
  /** Identifier of the embedding recipe that produced the vectors. */
  model: string;
  /** ISO-8601 timestamp of when the index was written. */
  createdAt: string;
  chunks: IndexedChunk[];
}
