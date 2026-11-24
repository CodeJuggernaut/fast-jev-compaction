import type {
  On,
  PluginOptions,
  Register,
  SessionMessage,
  ToolResultSummary,
  ToolUseSummary,
  TurnCompleteInput,
} from 'claude-code';

import { compact, reductionRatio, resolveOptions } from '../src/compact.js';
import { buildJevRequest, DEFAULT_MODEL, parseJevResponse } from '../src/request.js';
import type {
  CompactOptions,
  CompactResult,
  JevAsker,
  Message,
  ToolResult,
  ToolUse,
} from '../src/types.js';

const HOOK_DEFAULTS = {
  compactAtPercent: 60,
  model: DEFAULT_MODEL,
};

export type HookFetchInit = {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
};

export type HookFetchResponse = {
  status: number;
  ok: boolean;
  text: string;
};

/** The shape of `$.http.fetch`, so the hook can be driven without an engine. */
export type HookFetch = (url: string, init?: HookFetchInit) => Promise<HookFetchResponse>;

export type HookConfig = CompactOptions & {
  apiKey?: string;
  compactAtPercent: number;
  model: string;
};

function optionNumber(options: PluginOptions, key: string, fallback: number): number {
  const value = options[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function optionString(options: PluginOptions, key: string): string | undefined {
  const value = options[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

/** Reads the plugin's `userConfig` values; anything missing takes the defaults. */
export function resolveHookConfig(options: PluginOptions): HookConfig {
  const numbers: Partial<Omit<CompactOptions, 'goal'>> = {};
  for (const key of [
    'keepThreshold',
    'preserveRecentMessages',
    'maxStateTokens',
    'maxRequestTokens',
    'truncateHeadChars',
  ] as const) {
    const value = options[key];
    if (typeof value === 'number' && Number.isFinite(value)) numbers[key] = value;
  }
  const config: HookConfig = {
    ...numbers,
    compactAtPercent: optionNumber(options, 'compactAtPercent', HOOK_DEFAULTS.compactAtPercent),
    model: optionString(options, 'model') ?? HOOK_DEFAULTS.model,
  };
  const apiKey = optionString(options, 'apiKey');
  if (apiKey) config.apiKey = apiKey;
  const goal = optionString(options, 'goal');
  if (goal) config.goal = goal;
  return config;
}

/** A `JevAsker` over the engine's `$.http.fetch`. */
export function jevAsker(fetchFn: HookFetch, apiKey: string, model: string): JevAsker {
  return {
    async ask(state, questions) {
      const request = buildJevRequest({ apiKey, model }, state, questions);
      const response = await fetchFn(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.body,
      });
      return parseJevResponse(response.status, response.ok, response.text);
    },
  };
}

function toolUseSummary(tool: ToolUse): ToolUseSummary {
  const summary: ToolUseSummary = {
    tool_use_id: tool.tool_use_id,
    tool: tool.tool,
    input: tool.input,
  };
  if (tool.text !== undefined) summary.text = tool.text;
  if (tool.isError) summary.isError = true;
  return summary;
}

function toolResultSummary(result: ToolResult): ToolResultSummary {
  return {
    tool_use_id: result.tool_use_id,
    text: result.text,
    isError: result.isError ?? false,
  };
}

/**
 * Maps the library's output back onto session messages. Whatever came back
 * unchanged (a message, a tool use, a tool result) is the engine's own object,
 * handle included; anything rebuilt is a fresh message without a handle, so the
 * engine takes the edited content instead of its original.
 */
export function toSessionMessages(
  input: readonly SessionMessage[],
  output: readonly Message[],
): SessionMessage[] {
  const messages = new Map<Message, SessionMessage>();
  const uses = new Map<ToolUse, ToolUseSummary>();
  const results = new Map<ToolResult, ToolResultSummary>();
  for (const message of input) {
    messages.set(message, message);
    for (const tool of message.toolUses) uses.set(tool, tool);
    for (const result of message.toolResults ?? []) results.set(result, result);
  }
  // A message kept with its handle comes back whole, with every note the engine stapled to it. Only the last
  // keeps its handle, so those notes ride once; the rest are rebuilt, which also drops images in old tool results.
  return output.map((message, index) => {
    const own = messages.get(message);
    if (own && index === output.length - 1) return own;
    const rebuilt: SessionMessage = {
      role: message.role,
      text: message.text,
      toolUses: message.toolUses.map((tool) => uses.get(tool) ?? toolUseSummary(tool)),
    };
    if (message.toolResults && message.toolResults.length > 0) {
      rebuilt.toolResults = message.toolResults.map(
        (result) => results.get(result) ?? toolResultSummary(result),
      );
    }
    return rebuilt;
  });
}

export type SessionCompaction = {
  result: CompactResult;
  messages: SessionMessage[];
  /** Why Jev could not judge, when the no-Jev path ran instead. */
  jevError?: string;
};

/**
 * Runs the library over a session transcript. When the key is missing or Jev fails (TypeSafe answered 402 from
 * 2026-11-12), it runs again without Jev: older tool calls are dropped and the chat text still stays word for word.
 */
export async function compactSession(
  messages: readonly SessionMessage[],
  config: HookConfig,
  fetchFn: HookFetch,
  external?: (messages: readonly SessionMessage[]) => Promise<CompactResult>,
): Promise<SessionCompaction> {
  if (external) {
    try {
      const result = await external(messages);
      return { result, messages: toSessionMessages(messages, result.messages) };
    } catch {
      // The shared CLI is the live copy of the engine; when it cannot run, the bundled engine below does the job.
    }
  }
  try {
    if (!config.apiKey) throw new Error('TYPESAFE_API_KEY is not configured');
    const result = await compact(messages, jevAsker(fetchFn, config.apiKey, config.model), config);
    return { result, messages: toSessionMessages(messages, result.messages) };
  } catch (error) {
    const result = await compact(messages, null, config);
    return {
      result,
      messages: toSessionMessages(messages, result.messages),
      jevError: error instanceof Error ? error.message : String(error),
    };
  }
}

function percent(ratio: number): string {
  return `${Math.round(ratio * 100)}%`;
}

export function summarize(result: CompactResult): string {
  const { stats } = result;
  const parts = [
    stats.kept > 0 ? `${stats.kept} kept` : '',
    stats.resultsDropped > 0 ? `${stats.resultsDropped} results truncated` : '',
    stats.callsDropped > 0 ? `${stats.callsDropped} call_dropped` : '',
    stats.pinned > 0 ? `${stats.pinned} pinned` : '',
  ].filter(Boolean);
  return `${percent(reductionRatio(result))} reduction; ${
    parts.join(', ') || 'no tool calls'
  }; state ~${stats.stateTokens} tokens (${stats.stateStage}) in ${stats.requests} request(s)`;
}

const UI_LOG_MAX_CHARS = 4096;

export function decisionLog(result: CompactResult): string {
  return result.decisions
    .filter((d) => d.reason !== 'pinned')
    .map(
      (d) =>
        `${d.id}:${d.tool}:${d.action}/call=${d.keepCall.toFixed(2)}/result=${d.keepResult.toFixed(2)}`,
    )
    .join(' ');
}

export function decisionLogLines(
  result: CompactResult,
  maxChars: number = UI_LOG_MAX_CHARS,
): string[] {
  const entries = decisionLog(result).split(' ').filter(Boolean);
  if (entries.length === 0) return ['decisions: (none)'];
  const chunks: string[] = [];
  let current = '';
  for (const entry of entries) {
    const next = current ? `${current} ${entry}` : entry;
    if (current && next.length > maxChars - 24) {
      chunks.push(current);
      current = entry;
    } else current = next;
  }
  chunks.push(current);
  return chunks.map((chunk, index) =>
    chunks.length === 1
      ? `decisions: ${chunk}`
      : `decisions (${index + 1}/${chunks.length}): ${chunk}`,
  );
}

async function getApiKey(
  $: {
    env: { get: (name: string) => Promise<string | undefined> };
    settings: { read: () => Promise<Readonly<Record<string, unknown>>> };
  },
  config: HookConfig,
): Promise<string | undefined> {
  if (config.apiKey) return config.apiKey;
  const fromEnv = await $.env.get('TYPESAFE_API_KEY');
  if (fromEnv) return fromEnv;
  const settings = await $.settings.read();
  const env = settings['env'];
  if (env && typeof env === 'object') {
    const value = (env as Record<string, unknown>)['TYPESAFE_API_KEY'];
    if (typeof value === 'string' && value) return value;
  }
  return undefined;
}

/** Runs the shared CLI that npm run install:jev points at, so a new engine build applies to this live chat with no plugin reload. */
async function sharedCli($: {
  env: { get: (name: string) => Promise<string | undefined> };
  fs: { read: (path: string) => Promise<string> };
  process: { run: (argv: readonly string[], init?: { stdin?: string; timeoutMs?: number }) => Promise<{ exitCode: number; stdout: string; stderr: string }> };
}): Promise<((messages: readonly SessionMessage[]) => Promise<CompactResult>) | undefined> {
  const home = (await $.env.get('USERPROFILE')) ?? (await $.env.get('HOME'));
  const cli = home ? (await $.fs.read(`${home}/.config/code-ninja/jev-compact-cli.txt`).catch(() => '')).trim() : '';
  if (!cli) return undefined;
  return async (messages) => {
    const plain = messages.map(({ role, text, toolUses, toolResults }) => ({ role, text, toolUses, toolResults }));
    const run = await $.process.run(['node', cli], { stdin: JSON.stringify(plain), timeoutMs: 120_000 });
    if (run.exitCode !== 0) throw new Error(run.stderr.slice(0, 200));
    return JSON.parse(run.stdout) as CompactResult;
  };
}

function notify(
  $: {
    ui: {
      log: (text: string) => void;
      toast: (text: string, options?: { timeoutMs?: number }) => void;
    };
  },
  text: string,
): void {
  $.ui.log(text);
  $.ui.toast(text, { timeoutMs: 15_000 });
}

/**
 * Every message back word for word, tool data included. The answer when pruning fails or runs out of time: the
 * built-in summary never replaces the conversation (owner, 2026-11-24: "I don't want summaries").
 */
function keepEverything(messages: readonly SessionMessage[]): { messages: SessionMessage[] } {
  return { messages: toSessionMessages(messages, messages) };
}

export const register: Register = (on: On, options: PluginOptions) => {
  const configured = resolveHookConfig(options);
  let compacting = false;

  // Never calls next(event): below it, core's built-in summary would replace the conversation.
  on('session.compact', async ($, event) => {
    try {
      const config = { ...configured, apiKey: await getApiKey($, configured) };
      const { result, messages, jevError } = await compactSession(event.messages, config, async (url, init) => {
        const response = await $.http.fetch(url, init);
        return { status: response.status, ok: response.ok, text: response.text };
      }, await sharedCli($));
      for (const line of decisionLogLines(result)) $.ui.log(line);
      const withoutJev = jevError ? `; without Jev: ${jevError}` : '';
      notify(
        $,
        `kept ${messages.length}/${event.messages.length} messages, no summary (${summarize(result)}${withoutJev})`,
      );
      return { messages };
    } catch (error) {
      notify(
        $,
        `kept all ${event.messages.length} messages, no summary (pruning failed: ${error instanceof Error ? error.message : String(error)})`,
      );
      return keepEverything(event.messages);
    }
  }).catch(($, event, next) => {
    // Runs when the hook above overran its time budget or failed in a way it could not catch itself.
    notify(
      $,
      `kept all ${event.messages.length} messages, no summary (hook ${next.error.kind}${next.error.message ? `: ${next.error.message}` : ''})`,
    );
    return keepEverything(event.messages);
  });

  on('turn.complete', async ($, event: TurnCompleteInput, next) => {
    if (compacting) return next(event);
    try {
      const { context } = await $.session.usage();
      await $.fs.write(`${(await $.env.get('USERPROFILE')) ?? ''}/.config/code-ninja/compact-trigger.json`, JSON.stringify({ at: new Date().toISOString(), percent: context.percent ?? null, tokens: context.tokens ?? null, window: context.window, threshold: configured.compactAtPercent }));
      if ((context.percent ?? 0) < configured.compactAtPercent) return next(event);
      compacting = true;
      await $.session.compact();
    } catch (error) {
      $.ui.log(
        `auto-compact skipped (${error instanceof Error ? error.message : String(error)})`,
      );
    } finally {
      compacting = false;
    }
    return next(event);
  });
};

export { resolveOptions };
