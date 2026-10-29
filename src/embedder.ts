/** Dimensionality of the default hashing-trick embedding vectors. */
export const EMBEDDING_DIM = 256;

/** Options for {@link embed}. */
export interface EmbedOptions {
  /** Vector dimensionality; defaults to {@link EMBEDDING_DIM}. */
  dim?: number;
  /** Smallest n-gram size (inclusive); defaults to 1. */
  ngramMin?: number;
  /** Largest n-gram size (inclusive); defaults to 3. */
  ngramMax?: number;
}

/**
 * Tokenize text into lowercase alphanumeric runs.
 *
 * @example tokenize('Hello, World! RAG-2') // ['hello', 'world', 'rag', '2']
 */
export function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
}

/** 32-bit FNV-1a hash, returned as an unsigned integer. Deterministic and dependency-free. */
export function fnv1a(input: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

/** Build all token n-grams of sizes `min` through `max` (joined with single spaces). */
export function ngrams(tokens: string[], min: number, max: number): string[] {
  const grams: string[] = [];
  for (let n = Math.max(1, min); n <= max; n++) {
    for (let i = 0; i + n <= tokens.length; i++) {
      grams.push(tokens.slice(i, i + n).join(' '));
    }
  }
  return grams;
}

/** Return an L2-normalized copy of `vector`; a zero vector stays zero. */
export function normalize(vector: number[]): number[] {
  let squared = 0;
  for (const value of vector) squared += value * value;
  const norm = Math.sqrt(squared);
  if (norm === 0) return vector.map(() => 0);
  return vector.map((value) => value / norm);
}

/**
 * Embed text with the hashing trick: token n-grams are hashed (FNV-1a) into a
 * fixed-dimension vector via signed projection, then L2-normalized. The same
 * text always produces the same vector — no model, no network, no randomness.
 *
 * @returns A vector of length `dim` with unit L2 norm (all zeros for empty text).
 */
export function embed(text: string, options: EmbedOptions = {}): number[] {
  const dim = options.dim ?? EMBEDDING_DIM;
  if (!Number.isInteger(dim) || dim < 1) throw new Error(`Invalid embedding dimension: ${dim}`);
  const vector = new Array<number>(dim).fill(0);
  const grams = ngrams(tokenize(text), options.ngramMin ?? 1, options.ngramMax ?? 3);
  for (const gram of grams) {
    const hash = fnv1a(gram);
    const sign = (hash >>> 31) === 1 ? -1 : 1;
    vector[hash % dim] += sign;
  }
  return normalize(vector);
}

/** Cosine similarity between two vectors; 0 if either vector is zero. Throws on length mismatch. */
export function cosine(a: number[], b: number[]): number {
  if (a.length !== b.length) throw new Error(`Vector length mismatch: ${a.length} vs ${b.length}`);
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
