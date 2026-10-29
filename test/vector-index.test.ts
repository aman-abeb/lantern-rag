import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { chunkText } from '../src/chunker.js';
import { embed } from '../src/embedder.js';
import type { Chunk } from '../src/types.js';
import { VectorIndex } from '../src/vector-index.js';

const FIXTURE_URL = new URL('./fixtures/sample.md', import.meta.url);

function fixtureChunks(): Chunk[] {
  const text = readFileSync(FIXTURE_URL, 'utf8');
  return chunkText(text, 'test/fixtures/sample.md');
}

function buildFixtureIndex(): VectorIndex {
  const index = new VectorIndex();
  for (const chunk of fixtureChunks()) index.add(chunk, embed(chunk.text));
  return index;
}

describe('VectorIndex', () => {
  it('recalls the relevant fixture chunk in the top 3 (recall@3)', () => {
    const index = buildFixtureIndex();
    expect(index.size).toBeGreaterThanOrEqual(4);
    const results = index.search(embed('how does the hashing embedder work'), 3);
    expect(results).toHaveLength(3);
    expect(results.map((r) => r.chunk.text).join('\n')).toContain('embedder');
    for (let i = 1; i < results.length; i++) {
      expect(results[i - 1].score).toBeGreaterThanOrEqual(results[i].score);
    }
  });

  it('returns the embeddings section first for an embeddings-specific query', () => {
    const index = buildFixtureIndex();
    const results = index.search(embed('FNV-1a signed projection L2-normalized deterministic'), 1);
    expect(results[0].chunk.startLine).toBe(19);
    expect(results[0].chunk.file).toBe('test/fixtures/sample.md');
    expect(results[0].score).toBeGreaterThan(0);
  });

  it('returns at most k results, and all of them when k exceeds the index size', () => {
    const index = buildFixtureIndex();
    expect(index.search(embed('lantern'), 2)).toHaveLength(2);
    expect(index.search(embed('lantern'), 100)).toHaveLength(index.size);
  });

  it('ranks by cosine similarity with hand-built embeddings', () => {
    const index = new VectorIndex(2);
    index.add({ id: 'a', text: 'a', file: 'f', startLine: 1, endLine: 1, headingPath: [] }, [1, 0]);
    index.add({ id: 'b', text: 'b', file: 'f', startLine: 2, endLine: 2, headingPath: [] }, [0, 1]);
    const results = index.search([1, 0], 2);
    expect(results[0].chunk.id).toBe('a');
    expect(results[0].score).toBeCloseTo(1, 9);
    expect(results[1].score).toBeCloseTo(0, 9);
  });

  it('round-trips through index.json without changing search results', () => {
    const dir = mkdtempSync(join(tmpdir(), 'lantern-'));
    const path = join(dir, 'index.json');
    const index = buildFixtureIndex();
    index.save(path);
    const loaded = VectorIndex.load(path);
    expect(loaded.size).toBe(index.size);
    expect(loaded.dim).toBe(index.dim);
    const query = embed('what line metadata does a chunk keep');
    expect(loaded.search(query, 3)).toEqual(index.search(query, 3));
  });

  it('rejects embeddings of the wrong dimension', () => {
    const index = new VectorIndex(4);
    const chunk: Chunk = { id: 'x', text: 'x', file: 'f', startLine: 1, endLine: 1, headingPath: [] };
    expect(() => index.add(chunk, [1, 2])).toThrow();
    expect(() => index.search([1, 2, 3], 1)).toThrow();
  });
});
