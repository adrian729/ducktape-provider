# Agents

Agent SDKs and protocols, as of 2026-10-28. Question: can "agent" sit behind the same normalized interface as "model"?

## Backends

- Claude Agent SDK (Python, TS only): runs the Claude Code binary. `query(prompt, options)`, `SDKMessage` stream (`assistant`, `user`, `result` with usage/cost, `system` subtypes init/compact_boundary/api_retry/task_*, `stream_event`). Content = Messages API blocks. `canUseTool` callback, 30+ hooks, in-process MCP tools, sessions as jsonl (resume/fork), `interrupt()`.
- Claude Code CLI from any language: `claude -p --output-format stream-json --input-format stream-json --verbose` (`--bare` unverified). NDJSON of SDKMessage. Interactive permissions use an undocumented `control_request` channel; documented alternatives: `--permission-prompt-tool` (host MCP tool), hooks, `--permission-mode` (`--permission-prompts none` unverified).
- Claude Managed Agents (beta, REST + SSE, any language): agents, environments, sessions, threads. Events `agent.message/tool_use/tool_result`, `session.status_idle {stop_reason: end_turn | requires_action | budget_reached}`, `session.error`, `span.*`. Blocking events answered by `user.tool_confirmation`, `user.custom_tool_result`.
- OpenAI Agents SDK: library loop; internal stream = Responses events. Approvals via `result.interruptions` + re-run.
- OpenAI Agents API (hosted; `/v1/agents/sessions`, `/v1/agents/environments`): not evaluated yet. Candidate backend next to Managed Agents.
- Codex: `codex exec --json` (one-shot JSONL: thread/turn/item events, items agent_message, reasoning, command_execution, file_change, mcp_tool_call, todo_list). `codex app-server` (stdio JSON-RPC, long-lived): thread/turn/item methods, steer and interrupt (method names `turn/steer`, `turn/interrupt` unverified), approval server requests for command/file/permissions, dynamic host tools, token usage. Schema generated per version.
- Gemini CLI: `gemini --acp` (native ACP) or `-p --output-format stream-json`. Replaced by Antigravity CLI (`agy`) for free tiers 2026-06; `agy` has stream-json, no native ACP.
- OpenCode: `opencode serve` (HTTP + SSE, OpenAPI at `/doc`), `opencode acp`. Server API in flux (v2 `/api`).
- Goose: `goose acp`. Uses external ACP agents (Claude Code, Codex) as providers: the exact pattern under consideration.

## Protocols

- ACP (Agent Client Protocol): JSON-RPC over stdio, agent as subprocess. `session/new|load|resume|list|close`, `session/prompt {content[]} -> {stopReason}`, `session/cancel`. Updates: `agent_message_chunk`, `agent_thought_chunk`, `tool_call {id, title, kind, status, content[diff|terminal|content], locations, rawInput/rawOutput}`, `tool_call_update`, `plan`, `usage_update`, `compaction_*`. `session/request_permission {options[allow_once|allow_always|reject_once|reject_always]}`. Client offers fs and terminal. Content = MCP blocks (text, image, audio, resource_link, resource). SDKs 1.0 (Rust, TS) 2026-06; Python, Kotlin, Java exist. ~48 agents native or via adapter (Claude, Codex via adapters). v2 draft 2026-07: multi-turn prompts, idle signal, fork unstable.
- A2A: opaque agent-to-agent delegation (tasks, artifacts, parts). No tool calls, no deltas. Wrong granularity.
- AG-UI: UI event protocol: Run/Step lifecycle, TextMessage*, ToolCall*, Reasoning*, StateDelta (JSON Patch), Subagent*, interrupt + resume. Good vocabulary reference.
- Vercel AI SDK UI stream: `text-*`, `reasoning-*`, `tool-input-*`, `tool-approval-request`, `tool-output-available`, `source-*`, `file`, `data-*`, `finish`, `error`.
- Vercel `@ai-sdk/harness` (experimental): separate `HarnessV1` spec: `doStart(sandbox, workdir, permissionMode, resumeFrom) -> session {doPromptTurn, doContinueTurn, doSuspendTurn, doStop, doCompact, doReadHistory}`; reuses LanguageModelV4 parts + `file-change`, compaction. Adapters: claude-code, codex, opencode, cursor, cline, copilot, acp.
- Microsoft Agent Framework: `AIAgent` above `IChatClient`; hosted and A2A agents as other `AIAgent` impls. Clean two-layer.
- Gemini Interactions API: vendor unified model-or-agent on one endpoint (`model` or `agent`), `background`, `previous_interaction_id`.
- MCP: tool side only; every agent accepts host MCP servers. Portable way to inject host tools into an agent.

## Vocabulary overlap

Common to models and agents: text delta, thinking delta, tool call start/args/result, usage, stop reason, error, compaction.

Agent-only: permission request (needs an answer), plan/todo, file diff, terminal output, subagent scoping, session lifecycle (new/resume/fork/close), modes/config, steer mid-turn.

## Assessment

- One vocabulary for content and core events: yes. One interface: no. Model = stateless function (messages in, one message out; caller owns history, runs tools). Agent = stateful process (prompt in, many model calls + agent-run tools out; tool call is an observation with status; permission requests need answers).
- Forcing agents into `chat(model, messages, tools)` leaks: messages ignored, tools need MCP bridging, system partly honored, permissions have nowhere to go. "Agent as smart model" facade works only for single prompt in, final text out (community `ai-sdk-provider-claude-code`, Goose).
- Rule: lifecycle ids + status updates, not more event types. ACP, Codex, OpenCode, AG-UI all converge on "item with id, then status and content patches".
- ACP as basis for the agent side: mostly yes. Gaps: no tool-arg deltas, no text block start/stop, usage is context size + cost only, no thinking signature round-trip, host tools only via MCP, single-turn prompt in v1, assumes editor client (fs, terminal).

## Proposed shape (v2)

- `Agent.new_session(cwd, mcp_servers, config) -> Session`; `Session.prompt(content[]) -> events`; `Session.respond(permission_id, choice)`; `cancel()`; `resume/list/fork` capability-gated.
- Events = model core events + `tool_call_update {id, status: pending|in_progress|completed|failed|cancelled, kind?, title?, input?, output[]}`, `permission_request {id, tool_call_id, options[]}`, `plan {entries[]}`, `terminal_output {tool_call_id, delta}`, `file_change`, `turn_stop`. Optional `parent_id` on every event for subagents.
- Backends: ACP client first (covers ~48 agents). Native later where fidelity matters: Codex app-server, Claude stream-json, Managed Agents (plain HTTP).
- Spawned CLIs own their credentials; the backend passes an allowlisted environment so an ambient API key never overrides the CLI login (see [[Auth model]]).
- Blocks to add to the shared model for agents: `resource_link`/`resource` (MCP), `diff {path, old, new}` as tool-output content.

## Unverified

Claude CLI `--bare` and `--permission-prompts none`; Codex app-server method names for steer and interrupt; OpenAI Agents API shape and event stream; ACP v2 draft status.
