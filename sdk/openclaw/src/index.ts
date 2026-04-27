/**
 * @kayba_ai/openclaw-tracing — OpenClaw plugin that captures full agent turns
 * (user message, LLM input/output with thinking, tool calls, final reply) and
 * emits one structured Kayba trace per turn with real wall-clock timing,
 * trace-level sessionId/userId, and folder tagging.
 *
 * Pairs with `@kayba_ai/tracing`. The trader plugin (or any other tool plugin)
 * can keep its existing `kayba.trace()` wrapping for tool-level spans — those
 * land in the same kayba folder and can be cross-referenced by `runId`.
 */

import kayba from "@kayba_ai/tracing";
import {
  startSpan as mlflowStartSpan,
  updateCurrentTrace,
  SpanStatusCode,
  SpanType,
} from "mlflow-tracing";

// ── Hook payload shapes (probed against openclaw 2026.4.24) ────────────

interface PluginApi {
  pluginConfig?: Record<string, unknown>;
  logger: { info: (m: string) => void; warn: (m: string) => void; error?: (m: string) => void };
  on: (event: string, handler: (event: unknown, ctx: unknown) => unknown) => void;
}

interface MessageReceivedEvent {
  from?: string;
  content?: string;
  timestamp?: number;
  messageId?: string;
  senderId?: string;
  sessionKey?: string;
  metadata?: Record<string, unknown>;
}

interface MessageReceivedCtx {
  channelId?: string;
  sessionKey?: string;
  messageId?: string;
  senderId?: string;
}

interface BeforeAgentStartEvent {
  prompt?: string;
  runId?: string;
}

interface BeforeAgentStartCtx {
  runId?: string;
  agentId?: string;
  sessionKey?: string;
  sessionId?: string;
  channelId?: string;
}

interface LlmInputEvent {
  runId: string;
  sessionId: string;
  provider: string;
  model: string;
  systemPrompt?: string;
  prompt: string;
  historyMessages: unknown[];
  imagesCount: number;
}

interface LlmOutputEvent {
  runId: string;
  sessionId: string;
  provider: string;
  model: string;
  resolvedRef?: string;
  harnessId?: string;
  assistantTexts: string[];
  lastAssistant?: {
    role?: string;
    content?: Array<{ type: string; text?: string; thinking?: string; name?: string; arguments?: unknown }>;
    usage?: unknown;
    stopReason?: string;
    timestamp?: string | number;
    responseId?: string;
  };
  usage?: { input?: number; output?: number; cacheRead?: number; cacheWrite?: number; total?: number };
}

interface AgentEndEvent {
  runId?: string;
  messages: unknown[];
  success: boolean;
  error?: string;
  durationMs?: number;
}

interface ConversationCtx {
  runId?: string;
  trace?: { traceId?: string; spanId?: string; traceFlags?: string };
  agentId?: string;
  sessionKey?: string;
  sessionId?: string;
  channelId?: string;
  trigger?: string;
}

// ── Per-turn state ─────────────────────────────────────────────────────

interface PendingTurn {
  runId?: string;
  sessionId?: string;
  sessionKey?: string;
  agentId?: string;
  channelId?: string;
  senderId?: string;
  userMessage?: string;
  startedAtMs: number;
  llmInputAtMs?: number;
  llmOutputAtMs?: number;
  endedAtMs?: number;
  llmIn?: LlmInputEvent;
  llmOut?: LlmOutputEvent;
  agentEnd?: AgentEndEvent;
  agentEndArrived?: boolean;
  finalized?: boolean;
}

const TURN_FINALIZE_DELAY_MS = 250;
const TURN_TIMEOUT_MS = 5 * 60 * 1000;

// ── Config ─────────────────────────────────────────────────────────────

interface PluginConfig {
  apiKey: string;
  baseUrl?: string;
  folder?: string;
  captureSystemPrompt: boolean;
  /**
   * "delta"  — only messages added since the previous turn for this sessionId (default).
   *            Traces stay ~5–10 KB regardless of conversation length.
   * "full"   — full historyMessages array on every turn. Bigger traces, no reconstruction needed.
   * "none"   — drop history entirely.
   */
  captureHistory: "delta" | "full" | "none";
  maxAttributeBytes: number;
  userField: "agentId" | "senderId";
}

function parseConfig(raw: unknown): PluginConfig {
  const obj = raw && typeof raw === "object" && !Array.isArray(raw) ? (raw as Record<string, unknown>) : {};
  let captureHistory: PluginConfig["captureHistory"] = "delta";
  if (obj.captureHistory === "full" || obj.captureHistory === true) captureHistory = "full";
  else if (obj.captureHistory === "none" || obj.captureHistory === false) captureHistory = "none";
  // Back-compat: systemPrompt should default to "first turn only" — capture once per session.
  // We model it by capturing only when captureSystemPrompt is true AND it's the session's first turn.
  return {
    apiKey: typeof obj.apiKey === "string" ? obj.apiKey : "",
    baseUrl: typeof obj.baseUrl === "string" ? obj.baseUrl : undefined,
    folder: typeof obj.folder === "string" ? obj.folder : undefined,
    captureSystemPrompt: obj.captureSystemPrompt !== false,
    captureHistory,
    maxAttributeBytes: typeof obj.maxAttributeBytes === "number" ? obj.maxAttributeBytes : 65536,
    userField: obj.userField === "senderId" ? "senderId" : "agentId",
  };
}

// ── Helpers ────────────────────────────────────────────────────────────

function truncate(value: unknown, maxBytes: number): unknown {
  if (typeof value !== "string") return value;
  if (value.length <= maxBytes) return value;
  return value.slice(0, maxBytes) + `…[truncated ${value.length - maxBytes} bytes]`;
}

/**
 * Recursively parse JSON-string fields back into structured values.
 *
 * OpenClaw's `historyMessages` is array<string>, where each string is a JSON
 * encoding of `{role, content, ...}`. The `content` field of those decoded
 * objects is *itself* a JSON-encoded array of `[{type, text|thinking|...}]`.
 * The same nesting shows up on `lastAssistant.content`, `usage`, `cost`, etc.
 *
 * If we ship those raw, mlflow JSON-stringifies the whole inputs/outputs blob
 * one more time on top, producing an unreadable wall of `\\\\\\\"`. Unwrapping
 * once before handoff yields clean, single-level JSON in the dashboard.
 *
 * Heuristic: a string is "wrapped JSON" if it starts with `{` or `[` and
 * `JSON.parse` succeeds. We cap recursion depth so a malicious payload can't
 * blow the stack.
 */
function unwrapJsonStrings(value: unknown, depth = 0): unknown {
  if (depth > 8) return value;
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (trimmed.length >= 2 && (trimmed[0] === "{" || trimmed[0] === "[")) {
      try {
        return unwrapJsonStrings(JSON.parse(trimmed), depth + 1);
      } catch {
        return value;
      }
    }
    return value;
  }
  if (Array.isArray(value)) return value.map((v) => unwrapJsonStrings(v, depth + 1));
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      out[k] = unwrapJsonStrings(v, depth + 1);
    }
    return out;
  }
  return value;
}

function safe<T>(fn: () => T, label: string, log: PluginApi["logger"]): T | undefined {
  try {
    return fn();
  } catch (err) {
    log.warn(`[kayba-tracing] ${label} failed: ${err instanceof Error ? err.message : String(err)}`);
    return undefined;
  }
}

function resolveUserId(turn: PendingTurn, field: PluginConfig["userField"]): string {
  return (field === "senderId" ? turn.senderId : turn.agentId) ?? "";
}

// ── Plugin entry ───────────────────────────────────────────────────────

export function register(api: PluginApi): void {
  const cfg = parseConfig(api.pluginConfig);
  if (!cfg.apiKey) {
    api.logger.error?.(
      `[kayba-tracing] missing required "apiKey" in plugin config; tracing disabled. ` +
        `Add plugins.entries.kayba-tracing.config.apiKey in openclaw.json.`,
    );
    return;
  }

  safe(
    () =>
      kayba.configure({
        apiKey: cfg.apiKey,
        baseUrl: cfg.baseUrl,
        folder: cfg.folder,
      }),
    "kayba.configure",
    api.logger,
  );

  if (!kayba.isConfigured()) {
    api.logger.warn(`[kayba-tracing] kayba SDK did not configure; tracing disabled`);
    return;
  }
  api.logger.info(`[kayba-tracing] configured (folder=${cfg.folder ?? "<none>"}, base=${cfg.baseUrl ?? "default"})`);

  // Per-turn state. Keyed by runId once known; before that, by `${sessionKey}:${messageId}` for
  // the brief window between message_received and before_agent_start.
  const turnsByRunId = new Map<string, PendingTurn>();
  const turnsByMessageKey = new Map<string, PendingTurn>();

  // Per-session bookkeeping for delta-mode history capture and one-shot system prompt.
  // Maps sessionId → number of historyMessages we've already shipped on prior turns.
  const sessionHistoryCursor = new Map<string, number>();
  const sessionsWithSystemPromptShipped = new Set<string>();

  function evictStaleTurns(): void {
    const cutoff = Date.now() - TURN_TIMEOUT_MS;
    for (const [k, t] of turnsByRunId) if (t.startedAtMs < cutoff) turnsByRunId.delete(k);
    for (const [k, t] of turnsByMessageKey) if (t.startedAtMs < cutoff) turnsByMessageKey.delete(k);
  }

  // ── message_received: open a turn keyed by sessionKey+messageId ─────

  api.on("message_received", (rawEvent, rawCtx) => {
    api.logger.info(`[kayba-tracing] hook: message_received`);
    const event = (rawEvent ?? {}) as MessageReceivedEvent;
    const ctx = (rawCtx ?? {}) as MessageReceivedCtx;
    const sessionKey = event.sessionKey ?? ctx.sessionKey ?? "";
    const messageId = event.messageId ?? ctx.messageId ?? "";
    if (!sessionKey || !messageId) return;
    const key = `${sessionKey}:${messageId}`;
    turnsByMessageKey.set(key, {
      sessionKey,
      channelId: ctx.channelId,
      senderId: event.senderId ?? ctx.senderId,
      userMessage: event.content,
      startedAtMs: typeof event.timestamp === "number" ? event.timestamp : Date.now(),
    });
    evictStaleTurns();
  });

  // ── before_agent_start: bind runId, promote to runId-keyed map ──────

  api.on("before_agent_start", (rawEvent, rawCtx) => {
    api.logger.info(`[kayba-tracing] hook: before_agent_start`);
    const event = (rawEvent ?? {}) as BeforeAgentStartEvent;
    const ctx = (rawCtx ?? {}) as BeforeAgentStartCtx;
    const runId = event.runId ?? ctx.runId;
    if (!runId) return;
    let turn: PendingTurn | undefined;
    if (ctx.sessionKey) {
      for (const [k, t] of turnsByMessageKey) {
        if (k.startsWith(ctx.sessionKey + ":") && !t.runId) {
          turn = t;
          turnsByMessageKey.delete(k);
          break;
        }
      }
    }
    if (!turn) {
      // No prior message_received (e.g. CLI-initiated agent run).
      turn = { startedAtMs: Date.now() };
    }
    turn.runId = runId;
    turn.sessionId = ctx.sessionId;
    turn.agentId = ctx.agentId;
    turn.channelId = turn.channelId ?? ctx.channelId;
    turnsByRunId.set(runId, turn);
  });

  // ── llm_input: stash the prompt + history, mark start time ──────────

  api.on("llm_input", (rawEvent, rawCtx) => {
    api.logger.info(`[kayba-tracing] hook: llm_input`);
    const event = (rawEvent ?? {}) as LlmInputEvent;
    const ctx = (rawCtx ?? {}) as ConversationCtx;
    const runId = event.runId ?? ctx.runId;
    if (!runId) return;
    let turn = turnsByRunId.get(runId);
    if (!turn) {
      turn = { runId, sessionId: event.sessionId, agentId: ctx.agentId, channelId: ctx.channelId, startedAtMs: Date.now() };
      turnsByRunId.set(runId, turn);
    }
    turn.llmIn = event;
    turn.llmInputAtMs = Date.now();
    turn.sessionId = turn.sessionId ?? event.sessionId;
  });

  // ── llm_output: stash response, mark end time ───────────────────────

  api.on("llm_output", (rawEvent, rawCtx) => {
    api.logger.info(`[kayba-tracing] hook: llm_output`);
    const event = (rawEvent ?? {}) as LlmOutputEvent;
    const ctx = (rawCtx ?? {}) as ConversationCtx;
    const runId = event.runId ?? ctx.runId;
    if (!runId) return;
    const turn = turnsByRunId.get(runId);
    if (!turn) return;
    turn.llmOut = event;
    turn.llmOutputAtMs = Date.now();
    if (turn.agentEndArrived) {
      void finalizeTurn(runId);
    }
  });

  // ── agent_end: defer finalize a tick to absorb any straggling llm_output ─

  api.on("agent_end", (rawEvent, rawCtx) => {
    api.logger.info(`[kayba-tracing] hook: agent_end`);
    const event = (rawEvent ?? {}) as AgentEndEvent;
    const ctx = (rawCtx ?? {}) as ConversationCtx;
    const runId = event.runId ?? ctx.runId;
    if (!runId) return;
    const turn = turnsByRunId.get(runId);
    if (!turn) return;
    turn.agentEnd = event;
    turn.endedAtMs = Date.now();
    turn.agentEndArrived = true;
    setTimeout(() => void finalizeTurn(runId), TURN_FINALIZE_DELAY_MS);
  });

  // ── Finalize: emit one trace per turn ───────────────────────────────

  async function finalizeTurn(runId: string): Promise<void> {
    const turn = turnsByRunId.get(runId);
    if (!turn || turn.finalized) {
      api.logger.info(`[kayba-tracing] finalize skipped (turn=${!!turn} finalized=${turn?.finalized})`);
      return;
    }
    turn.finalized = true;
    turnsByRunId.delete(runId);
    api.logger.info(`[kayba-tracing] finalizing turn runId=${runId} sess=${turn.sessionId} hasLlmIn=${!!turn.llmIn} hasLlmOut=${!!turn.llmOut}`);

    const sessionId = turn.sessionId ?? "";
    const userId = resolveUserId(turn, cfg.userField);
    const success = turn.agentEnd?.success ?? true;
    const realDurationMs = turn.agentEnd?.durationMs ?? (turn.endedAtMs ?? Date.now()) - turn.startedAtMs;

    // Set process-global session/user so the kayba SDK injects them into trace metadata.
    safe(() => kayba.setSession(sessionId || null), "setSession", api.logger);
    safe(() => kayba.setUser(userId || null), "setUser", api.logger);

    const traced = kayba.trace(
      async () => {
        // Trace-level metadata + previews. We're inside an active trace context here.
        safe(
          () =>
            updateCurrentTrace({
              metadata: {
                "openclaw.runId": runId,
                "openclaw.agentId": turn.agentId ?? "",
                "openclaw.channelId": turn.channelId ?? "",
                "openclaw.realDurationMs": String(realDurationMs),
              },
              requestPreview: turn.userMessage?.slice(0, 200),
              responsePreview: turn.llmOut?.assistantTexts?.[0]?.slice(0, 200),
            }),
          "updateCurrentTrace",
          api.logger,
        );

        // Nested llm.call span. Wall-clock duration is captured as an attribute
        // (Number.MAX_SAFE_INTEGER < Date.now() * 1_000_000, so passing startTimeNs
        // explicitly to mlflow corrupts the span — the OTel API expects nanos as a
        // number which JS can't represent past ~285k years from epoch).
        if (turn.llmIn || turn.llmOut) {
          const llmRealDurationMs =
            (turn.llmOutputAtMs ?? turn.endedAtMs ?? Date.now()) -
            (turn.llmInputAtMs ?? turn.startedAtMs);

          // Resolve which slice of historyMessages to ship.
          const fullHistory = turn.llmIn?.historyMessages ?? [];
          let historyToShip: unknown[] | undefined;
          let historyMode: "full" | "delta" | "none" = "none";
          let historySkipped = 0;
          if (cfg.captureHistory === "full" && fullHistory.length > 0) {
            historyToShip = fullHistory;
            historyMode = "full";
          } else if (cfg.captureHistory === "delta" && sessionId) {
            const cursor = sessionHistoryCursor.get(sessionId) ?? 0;
            historySkipped = Math.min(cursor, fullHistory.length);
            historyToShip = fullHistory.slice(historySkipped);
            historyMode = "delta";
            sessionHistoryCursor.set(sessionId, fullHistory.length);
          }

          // Capture systemPrompt only on the first turn of each session (it rarely changes).
          const shouldShipSystemPrompt =
            cfg.captureSystemPrompt &&
            !!turn.llmIn?.systemPrompt &&
            !!sessionId &&
            !sessionsWithSystemPromptShipped.has(sessionId);
          if (shouldShipSystemPrompt && sessionId) sessionsWithSystemPromptShipped.add(sessionId);

          const llmSpan = mlflowStartSpan({
            name: "llm.call",
            spanType: SpanType.LLM,
            attributes: {
              "openclaw.realDurationMs": String(llmRealDurationMs),
              "openclaw.startedAtMs": String(turn.llmInputAtMs ?? ""),
              "openclaw.endedAtMs": String(turn.llmOutputAtMs ?? ""),
              "openclaw.historyMode": historyMode,
              "openclaw.historySkipped": String(historySkipped),
              "openclaw.historyTotalLength": String(fullHistory.length),
            },
            inputs: {
              provider: turn.llmIn?.provider,
              model: turn.llmIn?.model,
              ...(shouldShipSystemPrompt
                ? { systemPrompt: truncate(turn.llmIn!.systemPrompt!, cfg.maxAttributeBytes) }
                : {}),
              prompt: truncate(turn.llmIn?.prompt, cfg.maxAttributeBytes),
              ...(historyToShip && historyToShip.length > 0
                ? { historyMessages: unwrapJsonStrings(historyToShip) }
                : {}),
              imagesCount: turn.llmIn?.imagesCount ?? 0,
            },
          });
          llmSpan.end({
            outputs: {
              assistantTexts: turn.llmOut?.assistantTexts,
              lastAssistant: unwrapJsonStrings(turn.llmOut?.lastAssistant),
              usage: unwrapJsonStrings(turn.llmOut?.usage),
              stopReason: turn.llmOut?.lastAssistant?.stopReason,
              resolvedRef: turn.llmOut?.resolvedRef,
            },
            status: success ? SpanStatusCode.OK : SpanStatusCode.ERROR,
          });
        }

        return {
          runId,
          sessionId,
          userId,
          channel: turn.channelId,
          senderId: turn.senderId,
          userMessage: turn.userMessage,
          assistantText: turn.llmOut?.assistantTexts?.[0],
          success,
          realDurationMs,
        };
      },
      {
        name: "agent.turn",
        spanType: SpanType.AGENT,
        attributes: {
          "openclaw.runId": runId,
          "openclaw.sessionId": sessionId,
          "openclaw.agentId": turn.agentId ?? "",
          "openclaw.channelId": turn.channelId ?? "",
          "openclaw.senderId": turn.senderId ?? "",
          "openclaw.success": String(success),
          "openclaw.realDurationMs": String(realDurationMs),
        },
      },
    );

    try {
      await traced();
      api.logger.info(`[kayba-tracing] emitted trace for runId=${runId} (real ${realDurationMs}ms)`);
    } catch (err) {
      api.logger.warn(`[kayba-tracing] trace emit failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
}

export default {
  id: "kayba-tracing",
  name: "Kayba Tracing",
  description: "Captures every OpenClaw agent turn and ships it as a Kayba trace.",
  register,
};
