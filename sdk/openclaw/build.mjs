import { build } from "esbuild";
import { rmSync } from "fs";

rmSync("./dist", { recursive: true, force: true });

await build({
  entryPoints: ["./src/index.ts"],
  outdir: "./dist",
  format: "esm",
  platform: "node",
  target: "node22",
  bundle: true,
  external: ["@kayba_ai/tracing", "mlflow-tracing", "openclaw/plugin-sdk"],
});

console.log("Build complete → dist/");
