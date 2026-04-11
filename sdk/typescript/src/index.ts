/**
 * Kayba tracing SDK for TypeScript/Node.js.
 *
 * Thin wrapper around `mlflow-tracing` that configures auth and
 * injects Kayba-specific folder tags.
 *
 * @example
 * ```ts
 * import kayba from "@kayba_ai/tracing";
 *
 * kayba.configure({
 *   apiKey: process.env.KAYBA_API_KEY,
 *   folder: "my-project",
 * });
 *
 * const myAgent = kayba.trace(async (topic: string) => {
 *   const span = kayba.startSpan({ name: "retrieval" });
 *   // ... work ...
 *   span.end({ status: "OK" });
 *   return result;
 * }, { name: "research_agent" });
 * ```
 */

import {
  init,
  trace as mlflowTrace,
  startSpan as mlflowStartSpan,
  updateCurrentTrace,
  SpanType,
  type TraceOptions as MlflowTraceOptions,
  type SpanOptions as MlflowSpanOptions,
} from "mlflow-tracing";

// ── Constants ──────────────────────────────────────────────────────────

const DEFAULT_BASE_URL = "https://use.kayba.ai";
const MAX_FOLDER_LENGTH = 256;
const SAFE_FOLDER_RE = /[^a-zA-Z0-9 _\-/.]/g;
const HTML_TAG_RE = /<[^>]*>/g;

// ── Module state ───────────────────────────────────────────────────────

let _folder: string | null = null;
let _configured = false;

// ── Types ──────────────────────────────────────────────────────────────

export interface ConfigureOptions {
  /** Kayba API key. Falls back to `KAYBA_API_KEY` env var. */
  apiKey?: string;
  /** Kayba API base URL. Falls back to `KAYBA_API_URL` env var, then `https://use.kayba.ai`. */
  baseUrl?: string;
  /** Alias for `folder`. If both provided, `folder` takes precedence. */
  experiment?: string;
  /** Folder name for organizing traces in the Kayba dashboard. */
  folder?: string;
  /** MLflow experiment ID. Defaults to `"0"`. */
  experimentId?: string;
}

export interface TraceOptions {
  /** Custom span name. Defaults to the function name. */
  name?: string;
  /** Span type (e.g. SpanType.LLM, SpanType.AGENT). */
  spanType?: SpanType;
  /** Additional span attributes. */
  attributes?: Record<string, unknown>;
}

export interface StartSpanOptions {
  /** Span name. */
  name: string;
  /** Span type. */
  spanType?: SpanType;
  /** Input data to attach to the span. */
  inputs?: Record<string, unknown>;
}

// ── Internal helpers ───────────────────────────────────────────────────

function sanitizeFolder(name: string): string {
  let clean = name.replace(HTML_TAG_RE, "");
  clean = clean.replace(SAFE_FOLDER_RE, "");
  return clean.trim().slice(0, MAX_FOLDER_LENGTH);
}

function resolveFolder(
  folder: string | undefined,
  experiment: string | undefined,
): string | null {
  const raw = folder ?? experiment;
  if (!raw) return null;
  const sanitized = sanitizeFolder(raw);
  return sanitized || null;
}

function injectFolderTag(): void {
  if (_folder !== null) {
    try {
      updateCurrentTrace({ tags: { "kayba.folder": _folder } });
    } catch {
      // Silently ignore if no active trace context.
    }
  }
}

// ── Public API ─────────────────────────────────────────────────────────

/**
 * Configure Kayba tracing.
 *
 * Sets the MLflow tracking URI and authentication so that all
 * subsequent `trace` / `startSpan` calls export to Kayba.
 */
export function configure(options: ConfigureOptions = {}): void {
  const apiKey = options.apiKey || process.env.KAYBA_API_KEY || "";

  if (!apiKey) {
    throw new Error(
      "No API key provided. Pass apiKey or set the KAYBA_API_KEY environment variable.",
    );
  }

  const baseUrl =
    options.baseUrl || process.env.KAYBA_API_URL || DEFAULT_BASE_URL;

  const trackingUri = baseUrl.replace(/\/+$/, "") + "/api/mlflow";

  // Configure MLflow under the hood.
  process.env.MLFLOW_TRACKING_TOKEN = apiKey;
  process.env.MLFLOW_TRACKING_URI = trackingUri;

  const experimentId =
    options.experimentId || process.env.MLFLOW_EXPERIMENT_ID || "0";

  init({ trackingUri, experimentId });

  _folder = resolveFolder(options.folder, options.experiment);
  _configured = true;
}

/**
 * Change the target folder for subsequent traces.
 *
 * @param folder - Folder name, or `null` to clear (traces go to Unfiled).
 */
export function setFolder(folder: string | null): void {
  if (folder === null) {
    _folder = null;
  } else {
    const sanitized = sanitizeFolder(folder);
    _folder = sanitized || null;
  }
}

/** Return the currently configured folder, or `null`. */
export function getFolder(): string | null {
  return _folder;
}

/**
 * Wrap a function with tracing. The returned function generates a span
 * each time it is called, with automatic folder tagging.
 *
 * @example
 * ```ts
 * const myFunc = trace(async (input: string) => {
 *   return doWork(input);
 * }, { name: "my_func", spanType: SpanType.LLM });
 * ```
 */
export function trace<T extends (...args: any[]) => any>(
  fn: T,
  options: TraceOptions = {},
): T {
  // Wrap the function to inject the folder tag after execution.
  const fnWithTag = (...args: Parameters<T>): ReturnType<T> => {
    const result = fn(...args);

    // Handle async functions: inject tag after the promise resolves.
    if (result instanceof Promise) {
      return result.then((value: unknown) => {
        injectFolderTag();
        return value;
      }) as ReturnType<T>;
    }

    injectFolderTag();
    return result;
  };

  const mlflowOptions: MlflowTraceOptions = {};
  if (options.name) mlflowOptions.name = options.name;
  if (options.spanType) mlflowOptions.spanType = options.spanType;
  if (options.attributes)
    mlflowOptions.attributes = options.attributes as Record<string, any>;

  return mlflowTrace(fnWithTag as T, mlflowOptions);
}

/**
 * Create a span manually. Call `.end()` when done.
 *
 * @example
 * ```ts
 * const span = startSpan({ name: "retrieval", spanType: SpanType.TOOL });
 * span.setInputs({ query });
 * // ... work ...
 * span.end({ outputs: { result }, status: "OK" });
 * ```
 */
export function startSpan(options: StartSpanOptions) {
  const mlflowOptions: MlflowSpanOptions = {
    name: options.name,
  };
  if (options.spanType) mlflowOptions.spanType = options.spanType;
  if (options.inputs) mlflowOptions.inputs = options.inputs;

  const span = mlflowStartSpan(mlflowOptions);

  // Wrap .end() to inject the folder tag before closing.
  const originalEnd = span.end.bind(span);
  span.end = (endOptions?: Parameters<typeof span.end>[0]) => {
    injectFolderTag();
    return originalEnd(endOptions);
  };

  return span;
}

/** Returns whether `configure()` has been called. */
export function isConfigured(): boolean {
  return _configured;
}

// ── Re-export MLflow types for convenience ─────────────────────────────

export { SpanType } from "mlflow-tracing";
export type { LiveSpan, Span } from "mlflow-tracing";

// ── Default export ─────────────────────────────────────────────────────

const kayba = {
  configure,
  trace,
  startSpan,
  setFolder,
  getFolder,
  isConfigured,
  SpanType,
};

export default kayba;
