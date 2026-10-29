import { describe, expect, it } from 'vitest';
import { EMBEDDING_DIM, cosine, embed, normalize, tokenize } from '../src/embedder.js';

describe('embed', () => {
  it('produces a fixed-dimension vector by default', () => {
    expect(embed('hello lantern')).toHaveLength(EMBEDDING_DIM);
  });

  it('is deterministic across calls', () => {
    expect(embed('the quick brown fox')).toEqual(embed('the quick brown fox'));
  });

  it('is L2-normalized for non-empty text', () => {
    const v = embed('hashing trick embedder with n-grams');
    const norm = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
    expect(norm).toBeCloseTo(1, 6);
  });

  it('returns a zero vector for empty text', () => {
    const v = embed('   ');
    expect(v).toHaveLength(EMBEDDING_DIM);
    expect(v.every((x) => x === 0)).toBe(true);
  });

  it('respects a custom dimension', () => {
    expect(embed('hello', { dim: 64 })).toHaveLength(64);
  });

  it('ranks related text above unrelated text', () => {
    const query = embed('cosine similarity between embedding vectors');
    const related = cosine(query, embed('cosine similarity of vector embeddings'));
    const unrelated = cosine(query, embed('banana bread recipe with yeast and flour'));
    expect(related).toBeGreaterThan(unrelated);
  });
});

describe('tokenize', () => {
  it('lowercases and strips punctuation', () => {
    expect(tokenize('Hello, World! RAG-2')).toEqual(['hello', 'world', 'rag', '2']);
  });

  it('returns an empty array for text without alphanumerics', () => {
    expect(tokenize(' — … !!!')).toEqual([]);
  });
});

describe('normalize', () => {
  it('scales to unit norm and leaves zero vectors at zero', () => {
    const v = normalize([3, 4]);
    expect(v[0]).toBeCloseTo(0.6, 6);
    expect(v[1]).toBeCloseTo(0.8, 6);
    expect(normalize([0, 0])).toEqual([0, 0]);
  });
});

describe('cosine', () => {
  it('is 1 for identical vectors and 0 for orthogonal or empty vectors', () => {
    expect(cosine([1, 2, 3], [1, 2, 3])).toBeCloseTo(1, 9);
    expect(cosine([1, 0], [0, 1])).toBe(0);
    expect(cosine([0, 0], [1, 1])).toBe(0);
  });

  it('throws on length mismatch', () => {
    expect(() => cosine([1], [1, 2])).toThrow();
  });
});
