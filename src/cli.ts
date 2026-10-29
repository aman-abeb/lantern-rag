#!/usr/bin/env node
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { extname, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { chunkText } from './chunker.js';
import { EMBEDDING_DIM, embed } from './embedder.js';
import type { SearchResult } from './types.js';
import { VectorIndex } from './vector-index.js';

/** File extensions treated as ingestible text documents. */
const TEXT_EXTENSIONS = new Set(['.md', '.markdown', '.txt']);
const DEFAULT_INDEX_PATH = 'index.json';
const DEFAULT_TOP_K = 5;

const USAGE = `Lantern — offline-first RAG CLI

Usage:
  lantern ingest <dir> [--index <path>]           Chunk, embed, and index a directory of docs
  lantern query <text> [-k <n>] [--index <path>]  Retrieve the top-k chunks for a query
  lantern help                                    Show this message

Options:
  -k, --top-k <n>  Number of chunks to return (default: ${DEFAULT_TOP_K})
  --index <path>   Index file path (default: ${DEFAULT_INDEX_PATH})
`;

/**
 * Recursively collect ingestible text files under `dir`.
 * Hidden directories and `node_modules` are skipped; results are sorted for determinism.
 */
export function collectTextFiles(dir: string): string[] {
  const files: string[] = [];
  const walk = (current: string): void => {
    const entries = readdirSync(current, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      const full = join(current, entry.name);
      if (entry.isDirectory()) {
        if (entry.name.startsWith('.') || entry.name === 'node_modules') continue;
        walk(full);
      } else if (entry.isFile() && TEXT_EXTENSIONS.has(extname(entry.name).toLowerCase())) {
        files.push(full);
      }
    }
  };
  walk(dir);
  return files;
}

/** Summary of an ingest run. */
export interface IngestSummary {
  files: number;
  chunks: number;
  dim: number;
}

/**
 * Chunk and embed every text file under `dir`, then write the vector index to `indexPath`.
 * Chunk `file` labels are the paths as walked (e.g. `docs/guide.md`), so citations stay readable.
 */
export function ingestDirectory(dir: string, indexPath: string): IngestSummary {
  const index = new VectorIndex(EMBEDDING_DIM);
  const files = collectTextFiles(dir);
  for (const file of files) {
    const text = readFileSync(file, 'utf8');
    for (const chunk of chunkText(text, file)) {
      index.add(chunk, embed(chunk.text));
    }
  }
  index.save(indexPath);
  return { files: files.length, chunks: index.size, dim: EMBEDDING_DIM };
}

/** Load the index at `indexPath` and return the top-`k` chunks for `text`. */
export function queryIndex(indexPath: string, text: string, k: number): SearchResult[] {
  const index = VectorIndex.load(indexPath);
  return index.search(embed(text), k);
}

/** Render one search result as `[rank] score  file:start-end` plus a one-line snippet. */
export function formatResult(result: SearchResult, rank: number): string {
  const { chunk, score } = result;
  const flat = chunk.text.replace(/\s+/g, ' ').trim();
  const snippet = flat.length > 180 ? `${flat.slice(0, 179)}…` : flat;
  return `[${rank}] ${score.toFixed(3)}  ${chunk.file}:${chunk.startLine}-${chunk.endLine}\n    ${snippet}`;
}

interface ParsedArgs {
  positional: string[];
  k: number;
  index: string;
}

/** Parse `-k/--top-k` and `--index` flags; everything else is positional. */
function parseArgs(args: string[]): ParsedArgs {
  const positional: string[] = [];
  let k = DEFAULT_TOP_K;
  let index = DEFAULT_INDEX_PATH;
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '-k' || arg === '--top-k') {
      const raw = args[++i];
      const value = Number(raw);
      if (raw === undefined || !Number.isInteger(value) || value < 1) {
        throw new Error(`Invalid value for ${arg}: ${raw ?? '(missing)'}`);
      }
      k = value;
    } else if (arg === '--index') {
      const value = args[++i];
      if (value === undefined) throw new Error('--index requires a file path');
      index = value;
    } else if (arg.startsWith('-') && arg !== '-') {
      throw new Error(`Unknown flag: ${arg}`);
    } else {
      positional.push(arg);
    }
  }
  return { positional, k, index };
}

/** CLI entry point. Returns the process exit code. */
export function main(argv: string[] = process.argv.slice(2)): number {
  const [command, ...rest] = argv;
  try {
    if (command === 'ingest') {
      const { positional, index } = parseArgs(rest);
      const dir = positional[0];
      if (dir === undefined) {
        console.error('Usage: lantern ingest <dir> [--index <path>]');
        return 1;
      }
      if (!existsSync(dir) || !statSync(dir).isDirectory()) {
        console.error(`Not a directory: ${dir}`);
        return 1;
      }
      const summary = ingestDirectory(dir, index);
      console.log(`Ingested ${summary.files} file(s) → ${summary.chunks} chunk(s) (dim=${summary.dim})`);
      console.log(`Index written to ${index}`);
      return 0;
    }
    if (command === 'query') {
      const { positional, k, index } = parseArgs(rest);
      const text = positional.join(' ').trim();
      if (text.length === 0) {
        console.error('Usage: lantern query <text> [-k <n>] [--index <path>]');
        return 1;
      }
      if (!existsSync(index)) {
        console.error(`Index not found: ${index} (run 'lantern ingest <dir>' first)`);
        return 1;
      }
      const results = queryIndex(index, text, k);
      if (results.length === 0) {
        console.log('Index is empty — nothing to retrieve.');
        return 0;
      }
      for (let i = 0; i < results.length; i++) {
        console.log(formatResult(results[i], i + 1));
      }
      return 0;
    }
    console.log(USAGE);
    return command === undefined || command === 'help' || command === '--help' || command === '-h' ? 0 : 1;
  } catch (error) {
    console.error(`Error: ${error instanceof Error ? error.message : String(error)}`);
    return 1;
  }
}

const invokedAsScript = process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href;
if (invokedAsScript) {
  process.exit(main());
}
