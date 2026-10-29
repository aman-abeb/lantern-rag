import { createHash } from 'node:crypto';
import type { Chunk } from './types.js';

/** Options controlling how a document is split into chunks. */
export interface ChunkOptions {
  /** Maximum characters per chunk (approximate; a single longer line is kept whole). */
  maxChars?: number;
  /** Approximate number of trailing characters repeated at the start of the next chunk. */
  overlapChars?: number;
}

/** Default maximum chunk size in characters. */
export const DEFAULT_MAX_CHARS = 900;
/** Default overlap between consecutive chunks, in characters. */
export const DEFAULT_OVERLAP_CHARS = 150;

const HEADING_RE = /^(#{1,6})\s+(.+?)\s*$/;

interface Line {
  n: number;
  text: string;
}

interface Section {
  headingPath: string[];
  lines: Line[];
}

/** Split a document into heading-delimited sections, tracking the heading path of each. */
function splitSections(text: string): Section[] {
  const lines = text.split(/\r?\n/).map((t, i) => ({ n: i + 1, text: t }));
  const sections: Section[] = [];
  const stack: { level: number; title: string }[] = [];
  let current: Section = { headingPath: [], lines: [] };
  for (const line of lines) {
    const match = HEADING_RE.exec(line.text);
    if (match) {
      if (current.lines.length > 0 || current.headingPath.length > 0) sections.push(current);
      const level = match[1].length;
      while (stack.length > 0 && stack[stack.length - 1].level >= level) stack.pop();
      stack.push({ level, title: match[2] });
      current = { headingPath: stack.map((s) => s.title), lines: [] };
    }
    current.lines.push(line);
  }
  if (current.lines.length > 0 || current.headingPath.length > 0) sections.push(current);
  return sections.filter((s) => s.lines.some((l) => l.text.trim().length > 0));
}

/** Stable chunk id derived from file, start line, and content. */
function chunkId(file: string, startLine: number, text: string): string {
  return createHash('sha1').update(`${file}:${startLine}:${text}`).digest('hex').slice(0, 16);
}

/** Cut one section into fixed-size windows with overlap, attaching line metadata. */
function chunkSection(section: Section, file: string, maxChars: number, overlapChars: number): Chunk[] {
  const chunks: Chunk[] = [];
  const lines = section.lines;
  let i = 0;
  while (i < lines.length) {
    let end = i;
    let len = 0;
    while (end < lines.length) {
      const lineLen = lines[end].text.length + 1;
      if (end > i && len + lineLen > maxChars) break;
      len += lineLen;
      end++;
    }
    const text = lines
      .slice(i, end)
      .map((l) => l.text)
      .join('\n')
      .trim();
    if (text.length > 0) {
      chunks.push({
        id: chunkId(file, lines[i].n, text),
        text,
        file,
        startLine: lines[i].n,
        endLine: lines[end - 1].n,
        headingPath: [...section.headingPath],
      });
    }
    if (end >= lines.length) break;
    // Next window starts far enough back that ~overlapChars of trailing text is repeated.
    let next = end;
    let acc = 0;
    while (next - 1 > i && acc + lines[next - 1].text.length + 1 <= overlapChars) {
      acc += lines[next - 1].text.length + 1;
      next--;
    }
    i = Math.max(next, i + 1);
  }
  return chunks;
}

/**
 * Chunk a Markdown or plain-text document.
 *
 * The document is split into sections at every ATX heading (`#` … `######`).
 * Sections longer than `maxChars` are cut into fixed-size windows that repeat
 * roughly `overlapChars` of trailing content. Every returned chunk carries the
 * source `file` and a 1-based inclusive `startLine`/`endLine` range.
 *
 * @param text - Full document text.
 * @param file - Path label stored on every chunk (used in citations).
 * @param options - Window size and overlap tuning.
 * @returns Chunks in document order; empty for blank input.
 */
export function chunkText(text: string, file: string, options: ChunkOptions = {}): Chunk[] {
  const maxChars = options.maxChars ?? DEFAULT_MAX_CHARS;
  const overlapChars = options.overlapChars ?? DEFAULT_OVERLAP_CHARS;
  if (!Number.isInteger(maxChars) || maxChars < 1) throw new Error(`maxChars must be a positive integer, got ${maxChars}`);
  if (!Number.isInteger(overlapChars) || overlapChars < 0 || overlapChars >= maxChars) {
    throw new Error(`overlapChars must be an integer in [0, maxChars), got ${overlapChars}`);
  }
  const chunks: Chunk[] = [];
  for (const section of splitSections(text)) {
    chunks.push(...chunkSection(section, file, maxChars, overlapChars));
  }
  return chunks;
}

/** Alias for {@link chunkText}, named for the primary use case. */
export const chunkMarkdown = chunkText;
