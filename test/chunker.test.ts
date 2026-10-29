import { describe, expect, it } from 'vitest';
import { chunkText } from '../src/chunker.js';

const doc = [
  '# Guide',
  '',
  'Lantern is an offline RAG tool.',
  '',
  '## Ingest',
  'Chunking splits documents on headings.',
  'Each chunk keeps file and line metadata.',
  '',
  '## Retrieve',
  'Brute force cosine similarity finds the nearest chunks.',
].join('\n');

describe('chunkText', () => {
  it('splits on headings and records the heading path of each section', () => {
    const chunks = chunkText(doc, 'guide.md');
    expect(chunks).toHaveLength(3);
    expect(chunks[0].headingPath).toEqual(['Guide']);
    expect(chunks[1].headingPath).toEqual(['Guide', 'Ingest']);
    expect(chunks[2].headingPath).toEqual(['Guide', 'Retrieve']);
  });

  it('attaches 1-based inclusive line ranges that match the source text', () => {
    const lines = doc.split('\n');
    const chunks = chunkText(doc, 'guide.md');
    expect(chunks.length).toBeGreaterThan(0);
    for (const chunk of chunks) {
      const slice = lines.slice(chunk.startLine - 1, chunk.endLine).join('\n').trim();
      expect(chunk.text).toBe(slice);
      expect(chunk.file).toBe('guide.md');
      expect(chunk.startLine).toBeGreaterThanOrEqual(1);
      expect(chunk.endLine).toBeGreaterThanOrEqual(chunk.startLine);
    }
  });

  it('produces a single chunk for documents under the size limit', () => {
    const chunks = chunkText('hello world', 'a.txt');
    expect(chunks).toHaveLength(1);
    expect(chunks[0].text).toBe('hello world');
    expect(chunks[0].startLine).toBe(1);
    expect(chunks[0].endLine).toBe(1);
    expect(chunks[0].headingPath).toEqual([]);
  });

  it('windows long sections with overlap and full line coverage', () => {
    const total = 40;
    const long = Array.from({ length: total }, (_, i) => `line ${String(i).padStart(3, '0')} ${'x'.repeat(30)}`).join('\n');
    const maxChars = 100;
    const overlapChars = 40;
    const chunks = chunkText(long, 'long.txt', { maxChars, overlapChars });
    expect(chunks.length).toBeGreaterThan(3);
    // Every source line is covered by at least one chunk.
    const covered = new Set<number>();
    for (const chunk of chunks) {
      for (let n = chunk.startLine; n <= chunk.endLine; n++) covered.add(n);
    }
    expect(covered.size).toBe(total);
    // Consecutive chunks overlap but always make progress.
    for (let i = 1; i < chunks.length; i++) {
      expect(chunks[i].startLine).toBeLessThanOrEqual(chunks[i - 1].endLine);
      expect(chunks[i].startLine).toBeGreaterThan(chunks[i - 1].startLine);
    }
    // Chunk sizes respect the window (allowing one over-long line).
    for (const chunk of chunks) {
      expect(chunk.text.length).toBeLessThanOrEqual(maxChars + 40);
    }
  });

  it('is deterministic: identical input yields identical chunk ids', () => {
    const a = chunkText(doc, 'guide.md').map((c) => c.id);
    const b = chunkText(doc, 'guide.md').map((c) => c.id);
    expect(a).toEqual(b);
    expect(new Set(a).size).toBe(a.length);
  });

  it('returns no chunks for blank input', () => {
    expect(chunkText('   \n\n  ', 'empty.md')).toEqual([]);
  });

  it('rejects invalid window options', () => {
    expect(() => chunkText('x', 'x.md', { maxChars: 0 })).toThrow();
    expect(() => chunkText('x', 'x.md', { maxChars: 10, overlapChars: 10 })).toThrow();
    expect(() => chunkText('x', 'x.md', { maxChars: 10, overlapChars: -1 })).toThrow();
  });
});
