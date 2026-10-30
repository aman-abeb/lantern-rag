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

/** Role of a message in a chat conversation. */
export type ChatRole = 'user' | 'assistant' | 'system';

/** A single message in a chat conversation. */
export interface ChatMessage {
  role: ChatRole;
  /** Message text. */
  content: string;
}

/** A `[n]` citation marker resolved to the source chunk it refers to. */
export interface Citation {
  /** 1-based citation number as printed in the answer text. */
  n: number;
  /** Stable id of the cited chunk. */
  chunkId: string;
  /** Source file of the cited chunk. */
  file: string;
  /** 1-based first line of the cited chunk (inclusive). */
  startLine: number;
  /** 1-based last line of the cited chunk (inclusive). */
  endLine: number;
}

/** An assistant answer: text with `[n]` markers plus the resolved citations. */
export interface ChatAnswer {
  /** Answer text containing `[n]` citation markers. */
  text: string;
  /** Citations resolved from the `[n]` markers, sorted by `n`. */
  citations: Citation[];
  /** Retrieved chunks used as context, in rank order (`[n]` refers to `sources[n - 1]`). */
  sources: SearchResult[];
}

/** Request sent to an LLM provider: the conversation plus the retrieved context. */
export interface CompletionRequest {
  /** Conversation so far, oldest first. */
  messages: ChatMessage[];
  /** Retrieved context chunks in rank order; `[n]` in the answer refers to `context[n - 1]`. */
  context: SearchResult[];
}

/** Response returned by an LLM provider. */
export interface CompletionResponse {
  /** Answer text; may contain `[n]` markers referring to the request's context. */
  text: string;
}
