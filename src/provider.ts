import { tokenize } from './embedder.js';
import type { CompletionRequest, CompletionResponse } from './types.js';

/**
 * Pluggable LLM backend. A provider receives the conversation so far plus the
 * retrieved context chunks and returns answer text that may contain `[n]`
 * markers referring to `context[n - 1]`.
 */
export interface LLMProvider {
  /** Short identifier printed by the CLI and in the eval report. */
  readonly name: string;
  /** Produce an answer for the given request. */
  complete(request: CompletionRequest): Promise<CompletionResponse>;
}

/**
 * Split text into trimmed sentence-like segments, breaking on sentence-ending
 * punctuation. Line breaks inside a paragraph are treated as plain spaces, so
 * hard-wrapped prose is never split mid-sentence; blank lines and heading lines
 * (starting with `#`) end the current segment, and heading-only segments are
 * dropped so answers read as prose.
 */
export function splitSentences(text: string): string[] {
  const paragraphs: string[] = [];
  let current = '';
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (trimmed.length === 0 || trimmed.startsWith('#')) {
      if (current.length > 0) paragraphs.push(current);
      current = '';
      continue;
    }
    current = current.length > 0 ? `${current} ${trimmed}` : trimmed;
  }
  if (current.length > 0) paragraphs.push(current);
  return paragraphs
    .flatMap((paragraph) => paragraph.split(/(?<=[.!?])\s+/))
    .map((segment) => segment.replace(/\s+/g, ' ').trim())
    .filter((segment) => segment.length > 0 && !segment.startsWith('#'));
}

/** Options for {@link MockProvider}. */
export interface MockProviderOptions {
  /** Maximum number of sentences composed into one answer. */
  maxSentences?: number;
}

const DEFAULT_MAX_SENTENCES = 3;

/**
 * Deterministic extractive provider used by default — no network, no API key.
 * Every sentence of the retrieved chunks is scored by token overlap with the
 * latest user message, and the best sentences are composed into an answer,
 * each followed by a `[n]` citation pointing at its chunk.
 */
export class MockProvider implements LLMProvider {
  readonly name = 'mock-extractive';
  private readonly maxSentences: number;

  constructor(options: MockProviderOptions = {}) {
    this.maxSentences = options.maxSentences ?? DEFAULT_MAX_SENTENCES;
  }

  async complete(request: CompletionRequest): Promise<CompletionResponse> {
    const context = request.context;
    if (context.length === 0) {
      return {
        text: "I don't have any indexed documents to answer from — run `lantern ingest <dir>` first.",
      };
    }
    const question = request.messages
      .filter((message) => message.role === 'user')
      .map((message) => message.content)
      .join(' ');
    const queryTokens = new Set(tokenize(question));
    const picked: { n: number; sentence: string }[] = [];
    for (let i = 0; i < context.length && picked.length < this.maxSentences; i++) {
      const sentences = splitSentences(context[i].chunk.text);
      if (sentences.length === 0) continue;
      let best = sentences[0];
      let bestScore = -1;
      for (const sentence of sentences) {
        const score = tokenize(sentence).filter((token) => queryTokens.has(token)).length;
        if (score > bestScore) {
          bestScore = score;
          best = sentence;
        }
      }
      if (bestScore > 0 || (i === 0 && picked.length === 0)) {
        picked.push({ n: i + 1, sentence: best });
      }
    }
    if (picked.length === 0) {
      // No token overlap at all: fall back to the first sentence of the top chunk.
      const first = splitSentences(context[0].chunk.text)[0] ?? context[0].chunk.text.trim();
      picked.push({ n: 1, sentence: first });
    }
    const text = picked.map((entry) => `${entry.sentence} [${entry.n}]`).join(' ');
    return { text };
  }
}

/** Options for {@link HttpProvider}. */
export interface HttpProviderOptions {
  /** OpenAI-style chat completions endpoint. */
  endpoint: string;
  /** Optional bearer token sent as `Authorization: Bearer <apiKey>`. */
  apiKey?: string;
  /** Optional model name sent in the request body. */
  model?: string;
}

/**
 * Optional real LLM backend over HTTP. It is only constructed when
 * `LANTERN_LLM_URL` is set (see {@link resolveProvider}), so the default
 * configuration never touches the network.
 */
export class HttpProvider implements LLMProvider {
  readonly name = 'http';

  constructor(private readonly options: HttpProviderOptions) {
    if (options.endpoint.trim().length === 0) throw new Error('HttpProvider requires a non-empty endpoint');
  }

  async complete(request: CompletionRequest): Promise<CompletionResponse> {
    const contextBlock = request.context
      .map((result, i) => `[${i + 1}] ${result.chunk.file}:${result.chunk.startLine}-${result.chunk.endLine}\n${result.chunk.text}`)
      .join('\n\n');
    const system =
      'You are Lantern, an offline-first RAG assistant. Answer the user question using only the ' +
      `context below, and cite the chunks you use as [n] markers (for example [1]).\n\nContext:\n${contextBlock || '(no context)'}`;
    const response = await fetch(this.options.endpoint, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        ...(this.options.apiKey ? { authorization: `Bearer ${this.options.apiKey}` } : {}),
      },
      body: JSON.stringify({
        model: this.options.model ?? 'lantern',
        messages: [{ role: 'system', content: system }, ...request.messages],
      }),
    });
    if (!response.ok) {
      throw new Error(`LLM endpoint responded with HTTP ${response.status}: ${await response.text()}`);
    }
    const data = (await response.json()) as { choices?: { message?: { content?: unknown } }[] };
    const text = data.choices?.[0]?.message?.content;
    if (typeof text !== 'string' || text.trim().length === 0) {
      throw new Error('LLM endpoint returned no message content');
    }
    return { text };
  }
}

/** Env var that enables the HTTP provider. */
export const LLM_URL_ENV = 'LANTERN_LLM_URL';
/** Env var with the optional API key for the HTTP provider. */
export const LLM_API_KEY_ENV = 'LANTERN_LLM_API_KEY';
/** Env var with the optional model name for the HTTP provider. */
export const LLM_MODEL_ENV = 'LANTERN_LLM_MODEL';

/**
 * Pick the active provider from environment variables only: an
 * {@link HttpProvider} when `LANTERN_LLM_URL` is set (with optional
 * `LANTERN_LLM_API_KEY` / `LANTERN_LLM_MODEL`), otherwise the offline
 * {@link MockProvider}. Never throws when the env vars are absent.
 */
export function resolveProvider(env: NodeJS.ProcessEnv = process.env): LLMProvider {
  const endpoint = env[LLM_URL_ENV];
  if (endpoint !== undefined && endpoint.trim().length > 0) {
    return new HttpProvider({
      endpoint,
      apiKey: env[LLM_API_KEY_ENV],
      model: env[LLM_MODEL_ENV],
    });
  }
  return new MockProvider();
}
