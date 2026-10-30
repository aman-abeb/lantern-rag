import { readFileSync } from 'node:fs';
import { buildChatAnswer } from './chat.js';
import { embed, tokenize } from './embedder.js';
import { splitSentences, type LLMProvider } from './provider.js';
import type { ChatAnswer, SearchResult } from './types.js';
import { VectorIndex } from './vector-index.js';

/** One line of a JSONL QA set. */
export interface QAItem {
  /** Stable identifier printed in the report. */
  id: string;
  /** The question asked to the RAG pipeline. */
  question: string;
  /**
   * Gold chunk references: either a chunk id or `file:startLine`, resolved
   * against the index at eval time.
   */
  gold: string[];
}

/** Per-question eval outcome. */
export interface EvalResult {
  id: string;
  question: string;
  /** Top-k retrieved chunks, in rank order. */
  retrieved: SearchResult[];
  /** Gold chunk ids resolved from the QA item. */
  goldIds: string[];
  /** Fraction of gold ids found in the top-k retrieved ids. */
  recallAtK: number;
  /** Mock-judge grounded score of the answer, in [0, 1]. */
  grounded: number;
  /** The answer text produced by the provider. */
  answer: string;
}

/** Aggregate eval report. */
export interface EvalReport {
  /** Name of the provider that produced the answers. */
  provider: string;
  /** Number of chunks retrieved per question. */
  k: number;
  /** Number of questions evaluated. */
  total: number;
  /** Mean recall@k across all questions. */
  recallAtK: number;
  /** Mean grounded-answer score across all questions. */
  grounded: number;
  results: EvalResult[];
}

/** Default number of chunks retrieved per question during eval. */
export const DEFAULT_EVAL_K = 3;
/** Minimum fraction of a sentence's tokens that must appear in the cited chunks to count as grounded. */
export const GROUNDED_TOKEN_THRESHOLD = 0.5;

/**
 * Parse a JSONL QA file; blank lines and `#` comments are skipped. The JSON
 * syntax of every line is checked first, so an invalid-JSON error (carrying its
 * line number) takes precedence over record-shape errors on earlier lines;
 * shape errors also carry the line number.
 */
export function parseQAJsonl(text: string): QAItem[] {
  const lines = text.split(/\r?\n/);
  const records: { lineNumber: number; value: unknown }[] = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.length === 0 || line.startsWith('#')) continue;
    try {
      records.push({ lineNumber: i + 1, value: JSON.parse(line) });
    } catch (error) {
      throw new Error(`Invalid JSON on line ${i + 1} of the QA file: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  const items: QAItem[] = [];
  for (const { lineNumber, value } of records) {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
      throw new Error(`Line ${lineNumber} of the QA file must be a JSON object`);
    }
    const record = value as Record<string, unknown>;
    if (typeof record.id !== 'string' || record.id.length === 0) {
      throw new Error(`Line ${lineNumber} of the QA file: 'id' must be a non-empty string`);
    }
    if (typeof record.question !== 'string' || record.question.length === 0) {
      throw new Error(`Line ${lineNumber} of the QA file: 'question' must be a non-empty string`);
    }
    if (!Array.isArray(record.gold) || record.gold.some((g) => typeof g !== 'string' || (g as string).length === 0)) {
      throw new Error(`Line ${lineNumber} of the QA file: 'gold' must be an array of non-empty strings`);
    }
    items.push({ id: record.id, question: record.question, gold: record.gold as string[] });
  }
  return items;
}

/** Load a QA set from a JSONL file on disk. */
export function loadQAFile(path: string): QAItem[] {
  return parseQAJsonl(readFileSync(path, 'utf8'));
}

/** Resolve a gold reference (chunk id or `file:startLine`) to a chunk id in `index`; null when unknown. */
export function resolveGoldRef(ref: string, index: VectorIndex): string | null {
  const trimmed = ref.trim();
  if (index.findById(trimmed) !== undefined) return trimmed;
  const match = /^(.*):(\d+)$/.exec(trimmed);
  if (match !== null) {
    const chunk = index.findByFileLine(match[1], Number(match[2]));
    if (chunk !== undefined) return chunk.id;
  }
  return null;
}

/** recall@k: fraction of `goldIds` present in the first `k` entries of `retrievedIds` (0 when there is no gold). */
export function recallAtK(retrievedIds: string[], goldIds: string[], k: number): number {
  if (goldIds.length === 0) return 0;
  const top = new Set(retrievedIds.slice(0, Math.max(0, k)));
  return goldIds.filter((id) => top.has(id)).length / goldIds.length;
}

/**
 * Deterministic mock judge for grounded answers: each sentence of the answer
 * (citation markers stripped) counts as grounded when at least
 * {@link GROUNDED_TOKEN_THRESHOLD} of its tokens appear in the union of the
 * cited chunks' tokens. Returns the grounded fraction in [0, 1]; 0 for an
 * empty answer.
 */
export function groundedAnswerScore(answer: ChatAnswer): number {
  const sentences = splitSentences(answer.text.replace(/\[\d+\]/g, ' '));
  if (sentences.length === 0) return 0;
  const support = new Set<string>();
  for (const source of answer.sources) {
    for (const token of tokenize(source.chunk.text)) support.add(token);
  }
  let grounded = 0;
  for (const sentence of sentences) {
    const tokens = tokenize(sentence);
    if (tokens.length === 0) continue;
    const hits = tokens.filter((token) => support.has(token)).length;
    if (hits / tokens.length >= GROUNDED_TOKEN_THRESHOLD) grounded++;
  }
  return grounded / sentences.length;
}

/** Run every QA item through retrieve + answer and compute recall@k and grounded-answer metrics. */
export async function runEval(
  index: VectorIndex,
  provider: LLMProvider,
  items: QAItem[],
  k: number = DEFAULT_EVAL_K,
): Promise<EvalReport> {
  const results: EvalResult[] = [];
  for (const item of items) {
    const retrieved = index.search(embed(item.question), k);
    const goldIds = item.gold
      .map((ref) => resolveGoldRef(ref, index))
      .filter((id): id is string => id !== null);
    const response = await provider.complete({
      messages: [{ role: 'user', content: item.question }],
      context: retrieved,
    });
    const answer = buildChatAnswer(response.text, retrieved);
    results.push({
      id: item.id,
      question: item.question,
      retrieved,
      goldIds,
      recallAtK: recallAtK(retrieved.map((r) => r.chunk.id), goldIds, k),
      grounded: groundedAnswerScore(answer),
      answer: answer.text,
    });
  }
  const mean = (values: number[]): number =>
    values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
  return {
    provider: provider.name,
    k,
    total: results.length,
    recallAtK: mean(results.map((r) => r.recallAtK)),
    grounded: mean(results.map((r) => r.grounded)),
    results,
  };
}

/** Render an {@link EvalReport} as a markdown document. */
export function renderMarkdownReport(report: EvalReport): string {
  const lines: string[] = [
    '# Lantern eval report',
    '',
    `- Provider: ${report.provider}`,
    `- Questions: ${report.total}`,
    `- recall@${report.k}: ${report.recallAtK.toFixed(3)}`,
    `- grounded-answer: ${report.grounded.toFixed(3)}`,
    '',
    `| ID | Question | recall@${report.k} | grounded | Top sources |`,
    '| --- | --- | ---: | ---: | --- |',
  ];
  for (const result of report.results) {
    const question = result.question.replace(/\|/g, '\\|');
    const sources = result.retrieved
      .slice(0, 3)
      .map((r, i) => `[${i + 1}] ${r.chunk.file}:${r.chunk.startLine}-${r.chunk.endLine}`)
      .join('<br>');
    lines.push(`| ${result.id} | ${question} | ${result.recallAtK.toFixed(2)} | ${result.grounded.toFixed(2)} | ${sources} |`);
  }
  lines.push('');
  return lines.join('\n');
}
