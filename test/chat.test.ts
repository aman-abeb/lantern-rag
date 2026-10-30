import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { buildChatAnswer, ChatSession, formatAnswer, handleChatLine } from '../src/chat.js';
import { chunkText } from '../src/chunker.js';
import { embed } from '../src/embedder.js';
import { HttpProvider, MockProvider, resolveProvider, type LLMProvider } from '../src/provider.js';
import type { CompletionRequest, SearchResult } from '../src/types.js';
import { VectorIndex } from '../src/vector-index.js';

const FIXTURE_URL = new URL('./fixtures/sample.md', import.meta.url);
const FIXTURE_FILE = 'test/fixtures/sample.md';

function fixtureIndex(): VectorIndex {
  const index = new VectorIndex();
  const text = readFileSync(FIXTURE_URL, 'utf8');
  for (const chunk of chunkText(text, FIXTURE_FILE)) index.add(chunk, embed(chunk.text));
  return index;
}

function topSources(): SearchResult[] {
  return fixtureIndex().search(embed('FNV-1a signed projection L2-normalized deterministic'), 3);
}

describe('buildChatAnswer / formatAnswer', () => {
  it('resolves [n] markers to file:startLine-endLine citations', () => {
    const sources = topSources();
    const answer = buildChatAnswer('Hashing n-grams. [1] Brute force cosine. [2] Out of range [9].', sources);
    expect(answer.citations.map((c) => c.n)).toEqual([1, 2]);
    const first = answer.citations[0];
    expect(first.file).toBe(FIXTURE_FILE);
    expect(first.startLine).toBe(19);
    expect(first.endLine).toBe(26);
    expect(first.chunkId).toBe(sources[0].chunk.id);
    const rendered = formatAnswer(answer);
    expect(rendered).toContain('[1] test/fixtures/sample.md:19-26');
    expect(rendered).toContain('Sources:');
    expect(rendered).toContain('[2] test/fixtures/sample.md:');
  });

  it('deduplicates repeated markers and ignores invalid ones', () => {
    const sources = topSources();
    const answer = buildChatAnswer('A [1] B [1] C [0] D [99]', sources);
    expect(answer.citations).toHaveLength(1);
    expect(answer.citations[0].n).toBe(1);
  });

  it('returns the bare text when there are no citations', () => {
    const answer = buildChatAnswer('No citations here.', []);
    expect(answer.citations).toEqual([]);
    expect(formatAnswer(answer)).toBe('No citations here.');
  });
});

describe('ChatSession', () => {
  it('answers with [n] citations resolved to source ranges and records history', async () => {
    const session = new ChatSession({ index: fixtureIndex(), provider: new MockProvider(), topK: 3 });
    const answer = await session.ask('How does the hashing embedder work?');
    expect(answer.text).toMatch(/\[1\]/);
    expect(answer.sources).toHaveLength(3);
    expect(answer.citations.length).toBeGreaterThan(0);
    const rendered = formatAnswer(answer);
    for (const citation of answer.citations) {
      expect(citation.file).toBe(FIXTURE_FILE);
      expect(citation.endLine).toBeGreaterThanOrEqual(citation.startLine);
      expect(rendered).toContain(`${citation.file}:${citation.startLine}-${citation.endLine}`);
    }
    expect(session.getHistory().map((m) => m.role)).toEqual(['user', 'assistant']);
    await session.ask('And retrieval?');
    expect(session.getHistory().map((m) => m.role)).toEqual(['user', 'assistant', 'user', 'assistant']);
    expect(session.getHistory()[2].content).toBe('And retrieval?');
  });

  it('is deterministic for the same question', async () => {
    const a = await new ChatSession({ index: fixtureIndex(), provider: new MockProvider() }).ask('What is the ingest pipeline?');
    const b = await new ChatSession({ index: fixtureIndex(), provider: new MockProvider() }).ask('What is the ingest pipeline?');
    expect(a.text).toBe(b.text);
  });

  it('passes conversation history and context to the provider', async () => {
    class RecordingProvider implements LLMProvider {
      readonly name = 'recording';
      readonly requests: CompletionRequest[] = [];
      async complete(request: CompletionRequest): Promise<{ text: string }> {
        this.requests.push(request);
        return { text: 'Recorded. [1]' };
      }
    }
    const provider = new RecordingProvider();
    const session = new ChatSession({ index: fixtureIndex(), provider, topK: 2 });
    await session.ask('first question');
    await session.ask('second question');
    expect(provider.requests).toHaveLength(2);
    expect(provider.requests[1].messages.map((m) => m.role)).toEqual(['user', 'assistant', 'user']);
    expect(provider.requests[1].messages[0].content).toBe('first question');
    expect(provider.requests[1].messages[2].content).toBe('second question');
    expect(provider.requests[1].context).toHaveLength(2);
  });
});

describe('handleChatLine', () => {
  it('returns null on /quit and /exit', async () => {
    const session = new ChatSession({ index: fixtureIndex(), provider: new MockProvider() });
    expect(await handleChatLine(session, '/quit')).toBeNull();
    expect(await handleChatLine(session, '  /exit  ')).toBeNull();
  });

  it('returns an empty string for blank input and a formatted answer otherwise', async () => {
    const session = new ChatSession({ index: fixtureIndex(), provider: new MockProvider() });
    expect(await handleChatLine(session, '   ')).toBe('');
    const output = await handleChatLine(session, 'How does the hashing embedder work?');
    expect(output).toContain('Sources:');
    expect(output).toMatch(/\[1\] test\/fixtures\/sample\.md:\d+-\d+/);
  });
});

describe('MockProvider', () => {
  it('composes an extractive answer with [n] citations from the context', async () => {
    const provider = new MockProvider();
    const response = await provider.complete({
      messages: [{ role: 'user', content: 'FNV-1a signed projection L2-normalized' }],
      context: topSources(),
    });
    expect(response.text).toContain('[1]');
    expect(response.text).toContain('FNV-1a');
  });

  it('answers gracefully with empty context', async () => {
    const response = await new MockProvider().complete({ messages: [{ role: 'user', content: 'hi' }], context: [] });
    expect(response.text).toContain('ingest');
    expect(response.text).not.toMatch(/\[1\]/);
  });
});

describe('resolveProvider', () => {
  it('falls back to the offline mock when no env vars are set', () => {
    const provider = resolveProvider({});
    expect(provider).toBeInstanceOf(MockProvider);
    expect(provider.name).toBe('mock-extractive');
  });

  it('uses the HTTP provider only when LANTERN_LLM_URL is set', () => {
    const provider = resolveProvider({
      LANTERN_LLM_URL: 'http://127.0.0.1:9/v1/chat/completions',
      LANTERN_LLM_API_KEY: 'test-key',
    });
    expect(provider).toBeInstanceOf(HttpProvider);
    expect(provider.name).toBe('http');
  });
});
