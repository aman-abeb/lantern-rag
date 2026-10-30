import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { buildChatAnswer } from '../src/chat.js';
import { chunkText } from '../src/chunker.js';
import { embed } from '../src/embedder.js';
import {
  groundedAnswerScore,
  parseQAJsonl,
  recallAtK,
  renderMarkdownReport,
  resolveGoldRef,
  runEval,
  type QAItem,
} from '../src/eval.js';
import { MockProvider } from '../src/provider.js';
import { VectorIndex } from '../src/vector-index.js';

const FIXTURE_URL = new URL('./fixtures/sample.md', import.meta.url);
const QA_URL = new URL('../eval/qa.jsonl', import.meta.url);
const FIXTURE_FILE = 'test/fixtures/sample.md';

function fixtureIndex(): VectorIndex {
  const index = new VectorIndex();
  const text = readFileSync(FIXTURE_URL, 'utf8');
  for (const chunk of chunkText(text, FIXTURE_FILE)) index.add(chunk, embed(chunk.text));
  return index;
}

describe('recallAtK', () => {
  it('computes the fraction of gold ids in the top k', () => {
    expect(recallAtK(['a', 'b', 'c'], ['a', 'x'], 3)).toBe(0.5);
    expect(recallAtK(['a', 'b', 'c'], ['a', 'b'], 2)).toBe(1);
    expect(recallAtK(['a', 'b', 'c'], ['c'], 1)).toBe(0);
    expect(recallAtK(['a'], [], 3)).toBe(0);
  });
});

describe('parseQAJsonl', () => {
  it('parses JSONL, skipping blanks and comments', () => {
    const items = parseQAJsonl(
      '# comment\n\n{"id":"q1","question":"Q?","gold":["f.md:1"]}\n{"id":"q2","question":"Q2","gold":["abc123"]}\n',
    );
    expect(items).toEqual([
      { id: 'q1', question: 'Q?', gold: ['f.md:1'] },
      { id: 'q2', question: 'Q2', gold: ['abc123'] },
    ]);
  });

  it('throws with the line number on invalid JSON or shape', () => {
    expect(() => parseQAJsonl('{"id":"q1"}\nnot json')).toThrow(/line 2/);
    expect(() => parseQAJsonl('{"id":"q1","question":"Q","gold":"nope"}')).toThrow(/gold/);
    expect(() => parseQAJsonl('{"question":"Q","gold":[]}')).toThrow(/id/);
  });
});

describe('resolveGoldRef', () => {
  it('resolves file:startLine refs and raw chunk ids', () => {
    const index = fixtureIndex();
    const id = resolveGoldRef(`${FIXTURE_FILE}:19`, index);
    expect(id).not.toBeNull();
    expect(index.findById(id as string)?.startLine).toBe(19);
    expect(resolveGoldRef(id as string, index)).toBe(id);
    expect(resolveGoldRef('missing.md:99', index)).toBeNull();
  });
});

describe('groundedAnswerScore', () => {
  it('scores verbatim extractive answers as fully grounded', () => {
    const sources = fixtureIndex().search(embed('FNV-1a signed projection L2-normalized deterministic'), 1);
    const answer = buildChatAnswer(
      'Each n-gram is hashed with FNV-1a into a fixed 256-dimensional vector using signed projection, and the result is L2-normalized. [1]',
      sources,
    );
    expect(groundedAnswerScore(answer)).toBe(1);
  });

  it('penalizes sentences unsupported by the cited chunks', () => {
    const sources = fixtureIndex().search(embed('FNV-1a signed projection L2-normalized deterministic'), 1);
    const answer = buildChatAnswer(
      'Each n-gram is hashed with FNV-1a into a fixed 256-dimensional vector using signed projection, and the result is L2-normalized. [1] Zebras wear striped pajamas on the moon. [1]',
      sources,
    );
    expect(groundedAnswerScore(answer)).toBe(0.5);
  });

  it('returns 0 for an empty answer', () => {
    expect(groundedAnswerScore(buildChatAnswer('', []))).toBe(0);
  });
});

describe('runEval', () => {
  it('computes recall@3 and grounded scores on a fixture question', async () => {
    const index = fixtureIndex();
    const items: QAItem[] = [
      { id: 'q1', question: 'FNV-1a signed projection L2-normalized deterministic', gold: [`${FIXTURE_FILE}:19`] },
    ];
    const report = await runEval(index, new MockProvider(), items, 3);
    expect(report.provider).toBe('mock-extractive');
    expect(report.total).toBe(1);
    expect(report.recallAtK).toBe(1);
    expect(report.grounded).toBe(1);
    expect(report.results[0].retrieved[0].chunk.startLine).toBe(19);
  });

  it('runs the bundled QA set end to end with metrics in [0, 1]', async () => {
    const items = parseQAJsonl(readFileSync(QA_URL, 'utf8'));
    expect(items.length).toBeGreaterThanOrEqual(4);
    const report = await runEval(fixtureIndex(), new MockProvider(), items, 3);
    expect(report.total).toBe(items.length);
    for (const result of report.results) {
      expect(result.recallAtK).toBeGreaterThanOrEqual(0);
      expect(result.recallAtK).toBeLessThanOrEqual(1);
      expect(result.grounded).toBeGreaterThanOrEqual(0);
      expect(result.grounded).toBeLessThanOrEqual(1);
      expect(result.goldIds.length).toBeGreaterThan(0);
    }
  });
});

describe('renderMarkdownReport', () => {
  it('emits a markdown report with recall@k and grounded-answer', async () => {
    const index = fixtureIndex();
    const items: QAItem[] = [
      { id: 'q1', question: 'FNV-1a signed projection L2-normalized deterministic', gold: [`${FIXTURE_FILE}:19`] },
    ];
    const report = await runEval(index, new MockProvider(), items, 3);
    const md = renderMarkdownReport(report);
    expect(md).toContain('# Lantern eval report');
    expect(md).toContain('- recall@3: 1.000');
    expect(md).toContain('- grounded-answer: 1.000');
    expect(md).toContain('| q1 | FNV-1a signed projection L2-normalized deterministic | 1.00 | 1.00 |');
    expect(md).toContain('[1] test/fixtures/sample.md:19-26');
  });
});
