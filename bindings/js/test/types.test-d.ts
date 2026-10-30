import { Provider, Stream, assistant, env, image, resolver, session, text, tool, toolResult, user } from "ducktape-provider";
import type { AuthStatus, Block, Capabilities, Event, Message, ModelInfo, Options, Response, SessionAuth, ToolDef } from "ducktape-provider";

const add: ToolDef = { type: "function", name: "add", description: "Adds", input_schema: { type: "object" } };
const p = new Provider("anthropic", { auth: [session("codex", { refresh: "memory" }), env("ANTHROPIC_API_KEY"), resolver(async () => ({ secret: "x" }))], default_options: { tool_choice: "auto" } });
const history: Message[] = [user("hi", image(new Uint8Array(1))), assistant(text("yo"))];
const r: Response = await p.chat(history, { tools: [add], options: { reasoning: "high", tool_choice: { name: "add" } } });
const t: string = r.text;
const calls = r.toolCalls;
history.push(tool(toolResult(calls[0]!.id, "4")));
const stream: Stream = p.stream(history, { model: "m" });
for await (const ev of stream) {
  const e: Event = ev;
  if (e.type === "text_delta") e.text.length;
  if (e.type === "finish") e.response.text.length;
}
const status: AuthStatus = await p.authStatus();
const caps: Capabilities = await p.capabilities("m");
const info: ModelInfo = await p.modelInfo("m");
const s: SessionAuth = session("anthropic_profile", { profile: "work" });
const o: Options = { max_output_tokens: 1, cache: "auto" };
const b: Block = { type: "custom", kind: "x" };
// @ts-expect-error wrong block type
const bad: Block = { type: "nope" };
// @ts-expect-error unknown option
new Provider("openai", { nope: 1 });
// @ts-expect-error refresh mode
session("codex", { refresh: "sometimes" });
export { t, status, caps, info, s, o, b, bad };
