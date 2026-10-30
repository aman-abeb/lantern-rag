import { embed } from './embedder.js';
import type { LLMProvider } from './provider.js';
import type { ChatAnswer, ChatMessage, Citation, CompletionRequest, SearchResult } from './types.js';
import { VectorIndex } from './vector-index.js';

/** Options for {@link ChatSession}. */
export interface ChatSessionOptions {
  /** Index used to retrieve context chunks for each question. */
  index: VectorIndex;
  /** Provider that composes the answer. */
  provider: LLMProvider;
  /** Number of chunks retrieved per question. */
  topK?: number;
}

const DEFAULT_TOP_K = 5;

/**
 * A stateful chat session: keeps the conversation history, retrieves the top-k
 * chunks for every question, asks the provider, and resolves the `[n]` markers
 * in the answer to `file:startLine-endLine` citations.
 */
export class ChatSession {
  private readonly history: ChatMessage[] = [];
  private readonly index: VectorIndex;
  private readonly provider: LLMProvider;
  private readonly topK: number;

  constructor(options: ChatSessionOptions) {
    this.index = options.index;
    this.provider = options.provider;
    this.topK = options.topK ?? DEFAULT_TOP_K;
  }

  /** Conversation history so far, oldest first (alternating user/assistant turns). */
  getHistory(): readonly ChatMessage[] {
    return [...this.history];
  }

  /**
   * Ask a question: retrieve the top-k chunks, call the provider with the
   * history plus context, resolve citations, and record the turn in history.
   */
  async ask(question: string): Promise<ChatAnswer> {
    const retrieved = this.index.search(embed(question), this.topK);
    const request: CompletionRequest = {
      messages: [...this.history, { role: 'user', content: question }],
      context: retrieved,
    };
    const response = await this.provider.complete(request);
    const answer = buildChatAnswer(response.text, retrieved);
    this.history.push({ role: 'user', content: question });
    this.history.push({ role: 'assistant', content: answer.text });
    return answer;
  }
}

/**
 * Resolve the `[n]` markers in `text` against `sources` (marker `[n]` refers to
 * `sources[n - 1]`). Out-of-range and repeated markers are ignored. The
 * returned answer carries citations sorted by `n`.
 */
export function buildChatAnswer(text: string, sources: SearchResult[]): ChatAnswer {
  const citations: Citation[] = [];
  const seen = new Set<number>();
  for (const match of text.matchAll(/\[(\d+)\]/g)) {
    const n = Number(match[1]);
    if (!Number.isInteger(n) || n < 1 || n > sources.length || seen.has(n)) continue;
    seen.add(n);
    const chunk = sources[n - 1].chunk;
    citations.push({
      n,
      chunkId: chunk.id,
      file: chunk.file,
      startLine: chunk.startLine,
      endLine: chunk.endLine,
    });
  }
  citations.sort((a, b) => a.n - b.n);
  return { text, citations, sources };
}

/**
 * Render an answer as the answer text followed by a `Sources:` footer listing
 * each citation as `[n] file:startLine-endLine`. Answers without citations are
 * returned as bare text.
 */
export function formatAnswer(answer: ChatAnswer): string {
  if (answer.citations.length === 0) return answer.text;
  const lines = answer.citations.map((c) => `[${c.n}] ${c.file}:${c.startLine}-${c.endLine}`);
  return `${answer.text}\n\nSources:\n${lines.join('\n')}`;
}

/**
 * Handle one line of chat input. Returns the text to print, an empty string
 * for blank input, or `null` when the user asked to quit (`/quit` or `/exit`).
 */
export async function handleChatLine(session: ChatSession, line: string): Promise<string | null> {
  const trimmed = line.trim();
  if (trimmed === '/quit' || trimmed === '/exit') return null;
  if (trimmed.length === 0) return '';
  return formatAnswer(await session.ask(trimmed));
}
