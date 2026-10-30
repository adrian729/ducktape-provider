# Auth model

Decision 10 of [[Design draft]], settled 2026-10-28; sub-decisions in Decided below. The library ships mechanisms; the consumer owns policy: every credential path below is a mechanism the consumer enables or disables in config. Builds on the Auth rows of [[Provider API matrix]]; the CLI-driven paths live in [[Agents]].

## Recommendation in one screen

- Split auth into two axes. **Kind** = how a credential is put on a request (`none`, `api_key`, `bearer`, `sigv4`). **Source** = where it comes from (`inline`, `env`, `resolver`, `session`, `exchange`). GCP and Azure are not kinds: they are exchange sources that yield a `bearer`.
- Precedence per provider: sources are tried in the literal order the consumer listed them (decision 18); the vendor's default env list applies only for the vendor's own `base_url`; `Auth/no_credential` names every source tried.
- Session discovery is off by default and enabled only by an explicit, ordered list in code (`source: [session: codex, session: claude_code]` or `Provider.from_session(...)`). No `auto`, no env switch.
- Core owns all auth logic as pure code: file-format parsing, expiry checks, refresh and exchange state machines, SigV4 and JWT signing, redaction, origin checks. Host does only raw I/O through effects: `NeedSecret`, `NeedHttp` (tagged `purpose: auth`), and two new effects, `NeedRead` and `NeedWrite`, for credential stores. The host passes `now` with every start and resume.
- Borrowed session stores (another tool's credentials) are read-only by default: on expiry, re-read the store, then fail with `Auth/credential_expired` and a hint naming the owning tool. In-memory refresh and write-back are explicit opt-ins with documented consequences.
- Full ambient cloud chains (AWS SSO and `credential_process`, `az login`, gcloud user login, managed identity variants) are delegated to each language's official identity library through `resolver` adapters. Core implements only the pure parts plus a small set of documented exchanges.

## Credential sources per vendor

Env var names and headers the official SDKs and CLIs use, as of 2026-10-28.

| Vendor | Secret env | Other env | Header on the wire | Notes |
|---|---|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY`; `ANTHROPIC_AUTH_TOKEN` | `ANTHROPIC_BASE_URL`, `ANTHROPIC_CUSTOM_HEADERS`, `ANTHROPIC_PROFILE`, `ANTHROPIC_CONFIG_DIR`, `ANTHROPIC_WORKSPACE_ID`, WIF set (`ANTHROPIC_FEDERATION_RULE_ID`, `ANTHROPIC_ORGANIZATION_ID`, `ANTHROPIC_SERVICE_ACCOUNT_ID`, `ANTHROPIC_IDENTITY_TOKEN[_FILE]`) | API key: `x-api-key` (Claude Code docs) or `Authorization: Bearer` (Python SDK source sends Bearer); auth token: Bearer | SDK order: constructor arg (total) > `ANTHROPIC_API_KEY` > `ANTHROPIC_AUTH_TOKEN` > `ANTHROPIC_PROFILE` > WIF env > active profile. Both env keys set: API key wins with a warning. Empty env var still occupies its slot. |
| OpenAI | `OPENAI_API_KEY` | `OPENAI_BASE_URL`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`, `OPENAI_ADMIN_KEY`, `OPENAI_WEBHOOK_SECRET` | `Authorization: Bearer`; `OpenAI-Organization`, `OpenAI-Project` | Admin key also Bearer. |
| Gemini API | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | none | `x-goog-api-key` (Bearer for OAuth) | `GOOGLE_API_KEY` wins when both are set. Google: never ship keys client-side. |
| Ollama | `OLLAMA_API_KEY` (ollama.com only) | `OLLAMA_HOST` (CLI and clients) | none locally; `Authorization: Bearer` on ollama.com (`x-api-key` alone rejected) | `ollama signin` makes the local server authenticate cloud models itself; the client sends nothing. |
| Bedrock | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (+ `AWS_SESSION_TOKEN`); `AWS_BEARER_TOKEN_BEDROCK` | `AWS_REGION`, `AWS_PROFILE`, shared files | SigV4 (service `bedrock-mantle` for the Messages endpoint); API key: Bearer on `bedrock-runtime`, `x-api-key` on Mantle per Anthropic docs | Short-term API key = `bedrock-api-key-` + base64 presigned SigV4 URL, max 12 h, minted locally. |
| Vertex / Google Cloud | `GOOGLE_APPLICATION_CREDENTIALS` (file path) | `GOOGLE_CLOUD_PROJECT`, ADC file | `Authorization: Bearer <access token>` | Claude on Vertex: `:rawPredict` / `:streamRawPredict`, `anthropic_version` in body. Anthropic-managed variant on `claude.googleapis.com` uses ADC bearer + `anthropic-version` header. |
| Azure / Foundry | `AZURE_OPENAI_API_KEY` (docs example; `OPENAI_API_KEY` also works on v1) | `OPENAI_BASE_URL`, `AZURE_TOKEN_CREDENTIALS` (narrows DefaultAzureCredential) | `api-key` (Responses v1); `api-key` or `x-api-key` (Claude); Entra: Bearer | Entra scope differs in docs: `https://ai.azure.com/.default` (OpenAI v1) vs `https://cognitiveservices.azure.com/.default` (Claude on Foundry). |

## Logged-in sessions: technical shape

What a library would read, send and refresh for each existing login. "Borrowed" = store owned by another tool.

| Session | Store (Linux / macOS / Windows) | Token shape | Requests go to | Refresh | Wire fit |
|---|---|---|---|---|---|
| Anthropic profile (`ant auth login`, Console OAuth) | `$ANTHROPIC_CONFIG_DIR` else `~/.config/anthropic` / same / `%APPDATA%\Anthropic`; `configs/<p>.json` (non-secret) + `credentials/<p>.json` (0600) | `access_token` `sk-ant-oat01-…`, `expires_at`, `refresh_token`; workspace-scoped | `api.anthropic.com`, `Authorization: Bearer`, `anthropic-beta: oauth-2025-04-20` | `refresh_token` grant at `{base_url}/v1/oauth/token`, `client_id` from the profile's auth block; SDK writes back by atomic replace, 0600; rotation supported | `anthropic.messages` unchanged. Documented, shared format meant for SDKs and the `ant` CLI. |
| Anthropic WIF | env set above or profile with `oidc_federation` | OIDC JWT in, `sk-ant-oat01-…` out, 60 s to 24 h | as above | `jwt-bearer` exchange at `/v1/oauth/token` with `anthropic-beta: oidc-federation-2026-04-01` (exchange only); `jti` assertions are single-use | `anthropic.messages` unchanged |
| Claude Code login | `~/.claude/.credentials.json` 0600 / Keychain (file fallback when locked) / `%USERPROFILE%\.claude\.credentials.json`; `CLAUDE_CONFIG_DIR` moves both | `claudeAiOauth {accessToken sk-ant-oat01-…, refreshToken, expiresAt ms, scopes, subscriptionType}` (unverified) | `api.anthropic.com`, Bearer + `oauth-2025-04-20` (unverified for this token type) | JSON `refresh_token` grant at `platform.claude.com/v1/oauth/token` (some reports: `console.anthropic.com`), public `client_id` (unverified); rotating; reported 400s when two processes refresh | `anthropic.messages`; server acceptance outside Claude Code is undocumented and has changed without notice (reports from 2026-01) |
| Claude Code long-lived token | `CLAUDE_CODE_OAUTH_TOKEN` env (from `claude setup-token`) | one-year OAuth token, inference only | as above | none; re-mint | as above |
| Codex login (ChatGPT) | `$CODEX_HOME/auth.json` (default `~/.codex`) or OS keyring service `Codex Auth`, account `cli` + first 16 hex of SHA-256 of canonical `CODEX_HOME`; mode set by `cli_auth_credentials_store` = `file`, `keyring`, `auto`, `ephemeral` | `auth_mode`, `OPENAI_API_KEY`, `tokens {id_token, access_token (JWT), refresh_token, account_id}`, `last_refresh` | `https://chatgpt.com/backend-api/codex` + `/responses`; `Authorization: Bearer`, `chatgpt-account-id` | JSON `refresh_token` grant at `https://auth.openai.com/oauth/token`, `client_id` `app_EMoamEEZ73f0CkXaXp7hrann`; Codex refreshes when `last_refresh` is older than 8 days or on 401, then writes back; errors `refresh_token_reused`, `_expired`, `_invalidated` show single-use rotation | `openai.responses` shape on a non-public backend; request constraints there are unverified |
| Codex login (API key mode) | same store | `OPENAI_API_KEY` field | `api.openai.com` | none | plain `api_key` |
| Sign in with ChatGPT, plan usage (library-owned login) | app-defined, e.g. `~/.config/<tool>/`, 0600, atomic writes | access 1 h, refresh 30 d rotating, scopes include `chatgpt.tokens.use.direct` | `https://api.openai.com/v1/responses`, Bearer | form `refresh_token` grant at `https://auth.openai.com/api/accounts/oauth/token` with issued `client_id` and `resource=https://api.openai.com/v1`; serialize refreshes; revoke via discovery `revocation_endpoint` | `openai.responses` with a dialect: `store: false`, `stream: true`, no system-role items, no `temperature`, `max_output_tokens`, `metadata`, `previous_response_id` and others; login = PKCE on loopback `127.0.0.1`, `client_id=dynamic_agent_client` on first registration, persistent `ext_agent_host_id` |
| Gemini CLI login (preset out, see Decided) | `~/.gemini/oauth_creds.json` or encrypted/keychain storage | Google installed-app OAuth (`access_token`, `refresh_token`, expiry) | `https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse`, body wraps `{model, project, request: <generateContent body>}`; project from `loadCodeAssist` / `onboardUser` | Google token endpoint with the CLI's embedded installed-app client id and secret | Not Interactions: needs a Code Assist codec over the `generateContent` shape, which [[Design draft]] rules out as a wire. Gemini CLI was replaced by Antigravity CLI for unpaid and Google One users on 2026-06-18; Antigravity's store is unverified. |
| Ollama signin | local server's own key | n/a | local server | n/a | `ollama.native`, `kind: none` |

The other way to use a CLI login is to run the CLI: Claude Agent SDK, `claude -p --output-format stream-json`, `codex exec --json`, `codex app-server`. The CLI then owns credentials, refresh and headers, so nothing is read or refreshed by us. That path belongs to the v2 agents layer, see [[Agents]]. Note for it: in `claude -p`, an `ANTHROPIC_API_KEY` in the environment is always used over the login, so the agents layer must control the child's environment.

## Cloud signing (wave 2): pure vs I/O

| Step | Pure (core) | I/O (host effect) |
|---|---|---|
| AWS SigV4 | canonical request, `SHA256(body)`, string to sign, `kDate → kRegion → kService → kSigning` HMAC chain, `Authorization` header, `x-amz-date`, `x-amz-security-token` in signed headers; inputs = access key, secret, session token, region, service, method, URL, headers, body, `now` | none |
| Bedrock short-term API key | presign a SigV4 URL and base64 it (12 h max, bounded by the source credentials' expiry) | none |
| AWS static / env creds | parse | `NeedSecret` (env read) |
| AWS shared files, SSO cache | parse INI / JSON | `NeedRead`; SSO refresh = `NeedHttp` |
| AWS web identity | build `AssumeRoleWithWebIdentity` request, parse response | `NeedRead` (token file), `NeedHttp` to STS |
| AWS container / IMDSv2 | build requests, parse | `NeedHttp` to container endpoint or `169.254.169.254` (PUT token, then GET) |
| AWS `credential_process`, IAM Identity Center login | none | subprocess or browser: delegate to `resolver` (official AWS SDK) |
| GCP service-account key | build and RS256-sign the JWT assertion | `NeedHttp` to `https://oauth2.googleapis.com/token` (jwt-bearer) |
| GCP ADC user (`authorized_user`) | refresh grant | `NeedRead` (ADC file: `~/.config/gcloud/application_default_credentials.json`, `%APPDATA%\gcloud\…`), `NeedHttp` |
| GCP metadata server | parse | `NeedHttp` GET `metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token`, header `Metadata-Flavor: Google`; server caches until 5 min before expiry |
| GCP workload / workforce federation | STS token-exchange request | `NeedRead`, `NeedHttp` to STS |
| Azure client secret / certificate / federated token | build `client_credentials` request; sign client assertion for cert | `NeedHttp` to Entra token endpoint; `NeedRead` for federated token file |
| Azure managed identity, `az login`, PowerShell, azd, broker | none | delegate to `resolver` (azure-identity); Microsoft itself recommends a specific credential over DefaultAzureCredential in production because env changes alter behavior |
| Anthropic WIF | build jwt-bearer body | `NeedRead` (identity token file, re-read every exchange), `NeedHttp` |

Consequences: core needs SHA-256, HMAC, RS256 (and ES256 if Azure certs or SigV4a are added) compiled for every target including wasm; the host passes `now` because signatures and expiry depend on time; clock skew is a real failure mode (signed requests carry a timestamp).

## Recommended auth model

### Config shape

```
endpoint.auth = {
  sources: [ Source, ... ],                                           # literal order; each source carries kind and placement (default from the wire; Anthropic API key = x-api-key)
  origins?: [ "https://api.anthropic.com" ],                         # where this credential may be sent; default = endpoint base_url origin
  extra_headers?: { "anthropic-beta": "oauth-2025-04-20" },          # set by session presets, not by the user
  refresh?: never | memory | write_back,                              # borrowed sessions only; default never
  dialect?: chatgpt_plan | ...                                        # request constraints a session imposes
}

Source = (all: kind?: none | api_key | bearer | sigv4, placement?: x-api-key | authorization | x-goog-api-key | api-key)
  inline   { secret_ref }                                             # handle filled through set_secret, never the secret in JSON
  env      { names: [ "ANTHROPIC_API_KEY", ... ] }                    # empty value = unset (warning)
  resolver { id }                                                     # host callback: ctx -> { secret | sigv4 creds, expires_at? }
  session  { preset: anthropic_profile | claude_code | claude_code_token_env | codex | chatgpt_plan, store?, client_id?, profile? }   # preset pins base_url, headers, dialect; gemini_cli: out (Decided)
  exchange { grant: refresh_token | jwt_bearer | client_credentials | sts_web_identity | gcp_metadata | aws_imds | aws_container, params }
```

- A `session` preset is data in core: store locators per OS, file format, kind, required headers, token endpoint, client id, refresh policy, dialect. Adding a preset is a core change plus fixtures, never a binding change.
- Constructor sugar: `auth="key"` = `[inline]`, `env(names)`, `session(preset, store?, client_id?)`, `resolver(fn)`, or a list of those. `login()` and `logout()` act on the provider's own session source.

### Precedence

1. Sources in literal list order; the first that yields a credential wins.
2. The vendor's default env list applies only when `base_url` is the vendor's own origin; a custom host requires explicit sources (decision 18).
3. Else `Auth/no_credential { tried: [labels] }`.

Every `Response.origin` carries a non-secret `auth_source` label (`env:OPENAI_API_KEY`, `session:codex`, `resolver`) so callers can see which credential paid for a call.

### Session discovery enablement

Off by default; explicit ordered list in code only. Rationale: a library embedded in a server must never pick up an operator's personal login from the home directory; env-driven ambient behavior is exactly what Microsoft warns about for DefaultAzureCredential; reading another tool's store may trigger OS keychain prompts; and the billing account behind a call must be a choice made in code. The app developer opts in once; the end user still does nothing beyond logging into the CLI, which keeps the zero-config experience.

### Effects

| Effect | Payload | Host does | Notes |
|---|---|---|---|
| `NeedSecret` | `{credential_ref, min_valid_until, force_refresh}` | answers from the binding's credential slot, which runs the core refresh sub-machine under a per-credential lock | single-flight: at most one resolver call or refresh per credential at a time; followers wait on the lock |
| `NeedHttp` | `{request, purpose: auth}` | sends it | auth-purpose requests: no body logging, no observer body, no retries outside core policy, never via a user proxy to loopback |
| `NeedRead` (new) | `{locator: file(path) or keychain(service, account) or env(name)}` | returns bytes or not-found | core parses; browser binding answers not-found for everything |
| `NeedWrite` (new) | `{locator, bytes, expect: hash of bytes read}` | atomic replace, mode 0600, compare-and-swap on `expect` | only when `refresh: write_back` or for library-owned stores |
| `NeedUserAction` (login) | `{url, code?, instructions}` | shows it however it wants: print, open browser, QR | no UI in the library |
| `NeedCallback` (login) | `{redirect_uri}` | listens on the loopback redirect or asks the user to paste the code; returns the callback URL | core verifies state and PKCE, exchanges the code via `NeedHttp`, stores via `NeedWrite` |
| `now` | passed on `start` and every `resume` | wall clock | keeps core pure; used for expiry and signing |

Refresh algorithm (core): advisory refresh 120 s before expiry, mandatory at 30 s (the Anthropic SDK's tuned values); a 401 triggers exactly one `force_refresh` and one retry; refresh errors split into permanent (`invalid_grant`, `*_reused`, `*_expired`, `*_invalidated`) and transient (network, 5xx) with a 5 s backoff so a token-endpoint outage is not hammered at request rate.

### Borrowed stores and write-back

| `refresh` | Behavior | Consequence |
|---|---|---|
| `never` (default) | use the stored access token; on expiry re-read the store (the owning tool may have refreshed), else fail `Auth/credential_expired { hint: "run codex" }` | owning tool stays healthy; calls fail when the user has not used that tool for a while |
| `memory` | refresh with the stored refresh token, keep result in process | with single-use rotation (Codex, Claude Code) this invalidates the owning tool's copy and logs it out |
| `write_back` | refresh, then `NeedWrite` with compare-and-swap on the file, preserving unknown fields; keychain write for keyring modes | races with the owning tool remain possible across processes (no shared lock protocol); format drift can corrupt the owner's store |

Library-owned stores (`chatgpt_plan`, a future own Anthropic profile login) always refresh and write back to their own path.

### Redaction rules

- Core type `Secret` (zeroize on drop; `Debug` and `Display` print `***`; no `Clone` into strings). Secrets enter through a dedicated `set_secret(handle, bytes)` boundary call, never inside the JSON config.
- Header names always redacted: `authorization`, `proxy-authorization`, `x-api-key`, `api-key`, `x-goog-api-key`, `x-amz-security-token`, `cookie`, `set-cookie`, `chatgpt-account-id`, `x-openai-actor-authorization`, plus any header the user marks secret.
- Query params: `key`, `access_token`, `code`, `id_token_hint`, `X-Amz-Credential`, `X-Amz-Signature`, `X-Amz-Security-Token`.
- JSON fields anywhere in auth traffic: every string except an allowlist (`type`, `version`, `expires_at`, `expires_in`, `token_type`, `scope`, `error`, `error_description`), following the Anthropic SDK's secret-by-default rule. Token-endpoint error bodies are reduced to status + `error` code because IdPs may echo credentials (Codex source comment).
- Value patterns scanned before any log, observer call or fixture write: `sk-ant-`, `sk-`, `sess-`, `AKIA`, `ASIA`, `ya29.`, `GOCSPX-`, `bedrock-api-key-`, JWT `eyJ…\.eyJ…`. Fixture recording fails closed on a hit.
- Observer hooks receive the redacted request; raw request access is not offered.

### Transport rules enforced by core

- Credentials are sent only to `origins` (default: the endpoint's own origin). Session presets pin vendor origins and token endpoints; neither `base_url` nor file contents nor tool override env vars (e.g. `CODEX_REFRESH_TOKEN_URL_OVERRIDE`) can redirect them.
- Scheme: `https`, or `http` only to a loopback IP literal (`127.0.0.0/8`, `::1`) with no proxy. Cloud metadata targets (`169.254.169.254`, `metadata.google.internal`) are reachable only from their `exchange` grants, never as a `base_url`.
- `NeedHttp` carries `follow_redirects: false` for authenticated requests; a redirect is surfaced to core, which never re-attaches auth across origins.
- User `headers` cannot set reserved auth headers or `host`; CR, LF and NUL in names or values are rejected before sending.

### Bindings

- Browser wasm: no `env`, `session` or file `exchange` sources (`NeedRead` returns not-found). `api_key` in a browser requires `dangerously_allow_browser: true`, mirroring the official TS SDKs, and then sends `anthropic-dangerous-direct-browser-access: true` on the Anthropic wire. Recommended browser pattern: `resolver` fetching short-lived tokens from the app's backend.
- C ABI: secrets in caller-owned buffers passed to `set_secret`; core copies and zeroizes its copy; the header documents that the caller zeroizes its own.
- Cloud adapters per language (optional extras): `resolver` implementations over boto3 / AWS SDK, google-auth, azure-identity. They cover the long tail of ambient chains with the vendor's own maintained code.

### Rationale

- Two axes keep the enum small: four ways to place a credential cover every wire in scope, while sources grow independently.
- Logic in core, bytes in host keeps the [[Design draft]] rule that glue carries no policy and lets the fixture suite test refresh, rotation and redaction once for every language.
- Opt-in sessions and `refresh: never` default make the borrowing paths available without surprising anyone: no ambient pickup, no silent logout of the owning tool.
- Delegating ambient cloud chains avoids re-implementing and maintaining security-sensitive code that each platform already ships.

## Security gaps

| Gap | Affected design | Mitigation |
|---|---|---|
| Secret lifetime in core memory | all kinds | `Secret` with zeroize on drop; short-lived derived values (SigV4 signing key) zeroized after use; allocator copies remain possible, documented |
| Secrets crossing FFI and JSON | all bindings | `set_secret` boundary call outside JSON; host strings (Python, JS, Java, .NET, Go) cannot be zeroized: minimize copies, document; outgoing headers must reach the host transport in clear, unavoidable |
| wasm linear memory | browser binding | zeroize inside wasm; any script on the page can read memory and network: browser keys are exposed by design, hence the `dangerously_allow_browser` gate |
| Secrets in vendor error bodies | errors | scrub error bodies; token endpoints reduced to status + error code; cap body size |
| Secrets in logs, debug dumps, observer | observer, `Debug` impls | redaction at the boundary; observer sees only redacted data |
| Secrets in panics and tracebacks | core, FFI | `catch_unwind` at every boundary; `Secret` never formats its value; Python binding wraps secret-bearing values so captured frame locals render redacted (Anthropic SDK practice) |
| Secrets in fixtures recorded from live traffic | conformance suite | recorder runs the full scrubber plus pattern scan and fails closed; CI re-scans committed fixtures |
| Resolver called per request | `resolver` | `expires_at` caching with advisory and mandatory windows; single-flight lock per credential; timeout; resolver exceptions mapped to `Auth/resolver_failed` without the message text |
| Refresh races, rotating refresh tokens | `session`, `exchange` | single-flight in process; compare-and-swap writes; `refresh: never` default for borrowed stores; permanent vs transient refresh errors |
| Writing into another tool's credential store | `write_back` | opt-in only; atomic replace 0600; preserve unknown fields; CAS; documented race with the owner |
| Credentials over plain http | all | https only; http only to loopback IP literal, no proxy |
| Cross-origin redirects leaking auth | transport | no automatic redirects on authenticated requests; core decides, never re-attaches auth cross-origin |
| Header injection via user headers | `headers`, `vendor` | reserved auth and host headers; CR/LF/NUL rejected |
| SSRF via user `base_url` | endpoint config | credential origin binding; metadata IPs rejected as `base_url`; session token endpoints pinned in core |
| Ambient credential pickup | `env`, `session` | sessions opt-in by explicit list; explicit sources are total; `auth_source` label on every response; empty env vars treated as unset with a warning |
| Env inheritance into subprocesses | agents layer | library never mutates env; agents layer spawns CLIs with an allowlisted env, so a stray `ANTHROPIC_API_KEY` cannot override the CLI login |
| Tool override env vars redirecting token endpoints | `session` | ignored; endpoints pinned in presets |
| PII in tokens and headers | logs | `id_token` claims, emails, `chatgpt-account-id` treated as secret |
| OS keychain prompts and access | `NeedRead` keychain | only when the preset is listed; document per OS; never loop on denial |
| Undocumented session endpoints changing | `claude_code`, `codex`, `gemini_cli` presets | presets versioned with fixtures; typed `Auth` errors carry `source`; failure never falls through silently to another source |
| Clock skew | SigV4, expiry | host `now`; skew detected from response `Date` and applied as an offset (common SDK practice, unverified per vendor) |
| Proxies and TLS interception | host transport | no option to disable certificate verification; custom CA bundle supported by the transport; proxy settings from host env, never for loopback |
| Supply chain of binding packages | packaging | trusted publishing with provenance, pinned toolchains, minimal deps, all credential code in core; see [[Build and packaging]] |

## Responsible use

Our terms carry a responsible-use clause: the consumer decides which credential mechanisms to enable and is solely responsible for complying with each vendor's terms for the credentials and sessions it uses.

## Decided (2026-10-28)

- Gemini CLI preset: out; a narrow exception may be added later.
- Login is part of the interface and UI-free: `login(method)` is a core state machine emitting `NeedUserAction {url, code?, instructions}` and `NeedCallback {redirect_uri}`; the host renders them (print, open browser, QR). Covers Sign in with ChatGPT and an Anthropic profile login; tokens obtained this way go to our own store, never into another tool's. `logout`, `auth_status`.
- Keychain reads in v1 via `NeedRead {store: keychain}` with a default implementation per binding (macOS Keychain, Windows Credential Manager, Linux Secret Service); our own store prefers the keychain, else a 0600 file.
- `anthropic_profile` login writes a profile in Anthropic's documented shared location; the client id a third party would use is unverified, so the preset ships only once a fixture proves the flow.
- Entropy for PKCE, state and `jti` comes from the host with every start and resume, next to `now`.

## Decided (2026-10-30, session sources shipped)

- Session presets `anthropic_profile`, `claude_code`, `claude_code_token_env` and `codex` are in core (`crates/core/src/session.rs`) with synthetic fixtures under `fixtures/wire/anthropic.messages/session_*` and `fixtures/wire/openai.responses/session_*`. `chatgpt_plan` and `login` fail with `unsupported{auth_source}` until a recorded fixture proves the flow; `logout` on a borrowed preset fails with the owning tool's command.
- `anthropic_profile` defaults to `refresh: write_back`, not `never`: the profile store is Anthropic's documented shared format and the official SDK itself refreshes and writes it back, so write-back is the expected behavior there. Only `user_oauth` profiles are read; federation profiles fail with `auth{auth_source}`. The config's `base_url` pins the endpoint; `workspace_id` goes out as `anthropic-workspace-id`.
- `claude_code` never refreshes (endpoint and client id unverified); on expiry it fails with `auth{credential_expired}` and the hint to run `claude`. `codex` defaults to `refresh: never`; `memory` and `write_back` use the public Codex client id at `auth.openai.com/oauth/token` and the written file keeps every field and updates `last_refresh`.
- A 401 on a session credential forces exactly one re-read of the store and, when refresh is allowed, one refresh, then one retry; a second 401 is the error.
- `auth_status()` never fails on an expired or unrefreshable session: it reports `source: session:<preset>` with the stale `expires_at`, so apps can tell "expired, run the tool" from "nothing configured" (`source: null`).
- Keychain reads (2026-10-30): core emits `NeedRead {keychain {service, account}}` for `codex`; the Python and JS hosts answer them through the platform's own tools, matching how the `keyring` crate 3.6 (which Codex uses) stores entries: macOS generic password with service and account (`security find-generic-password -s -a -w`), Linux Secret Service item with attributes `service` and `username` (`secret-tool lookup`), Windows generic credential with target name `<account>.<service>` and a UTF-16LE blob (`CredReadW` via ctypes in Python, via PowerShell `Add-Type` in Node). Both bindings take a `keychain` callable to replace the default. Only the Linux path has run locally (tool absent, answers not-found); macOS and Windows are written to the verified formats but not exercised.

## Verified / Unverified

Accessed 2026-10-28 unless noted; page dates in brackets where shown.

Verified:

- Claude Code authentication (storage paths, Keychain fallback, `CLAUDE_CONFIG_DIR`, precedence, `setup-token`, `-p` uses API key when present): https://code.claude.com/docs/en/authentication
- Anthropic profiles, `ant auth login`, credentials layout, precedence: https://platform.claude.com/docs/en/cli-sdks-libraries/cli/authentication
- WIF reference (token exchange, env vars, file layout, scopes, empty env var trap): https://platform.claude.com/docs/en/manage-claude/wif-reference
- Anthropic Python SDK credential chain, `oauth-2025-04-20` and `oidc-federation-2026-04-01` headers, 120 s / 30 s refresh windows, secret-by-default wrapping, Bearer for API keys: https://github.com/anthropics/anthropic-sdk-python (`src/anthropic/_client.py`, `src/anthropic/lib/credentials/_constants.py`, `_cache.py`, `_providers.py`, `_secrets.py`, main branch)
- Anthropic TS SDK `dangerouslyAllowBrowser` and `anthropic-dangerous-direct-browser-access`: https://github.com/anthropics/anthropic-sdk-typescript (`src/client.ts`, main)
- OpenAI Python SDK env vars and headers: https://github.com/openai/openai-python (`src/openai/_client.py`, main); TS `dangerouslyAllowBrowser`: https://github.com/openai/openai-node (`src/client.ts`, main)
- Codex auth storage, `cli_auth_credentials_store`, auth.json guidance: https://learn.chatgpt.com/docs/auth; CI/CD refresh mechanics (8 days, write-back, concurrency): https://learn.chatgpt.com/docs/auth/ci-cd-auth
- Codex source: `CHATGPT_CODEX_BASE_URL`, refresh URL, client id, rotation error codes, keyring service and key, `chatgpt-account-id`: https://github.com/openai/codex (`codex-rs/model-provider-info/src/lib.rs`, `codex-rs/login/src/auth/manager.rs`, `codex-rs/login/src/auth/storage.rs`, `codex-rs/login/src/token_data.rs`, `codex-rs/http-client/src/openai_headers.rs`; last commit to manager.rs 2026-10-27)
- Sign in with ChatGPT plan usage (registration, endpoints, lifetimes, refresh, Responses constraints, app-server config): https://developers.openai.com/siwc/llms.txt and pages under https://developers.openai.com/siwc/token-sharing-open-source
- Gemini API keys, env precedence, `x-goog-api-key` [2026-10-27]: https://ai.google.dev/gemini-api/docs/api-key
- Gemini CLI auth methods, replacement by Antigravity CLI on 2026-06-18 [2026-09-18]: https://geminicli.com/docs/get-started/authentication/
- Gemini CLI source: Code Assist endpoint and version, OAuth client constants, `oauth_creds.json`: https://github.com/google-gemini/gemini-cli (`packages/core/src/code_assist/server.ts` [last commit 2026-07-23], `oauth2.ts`, `packages/core/src/config/storage.ts`)
- Ollama cloud auth: https://docs.ollama.com/api/authentication
- AWS SigV4 signing steps: https://docs.aws.amazon.com/IAM/latest/UserGuide/create-signed-request.html; credential provider chain: https://docs.aws.amazon.com/sdkref/latest/guide/standardized-credentials.html
- Bedrock API keys: https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html; short-term key format: https://github.com/aws/aws-bedrock-token-generator-python (README)
- Claude in Amazon Bedrock (Mantle endpoint, `bedrock-mantle` signing name, bearer on `x-api-key`): https://platform.claude.com/docs/en/build-with-claude/claude-in-amazon-bedrock; legacy Bedrock: https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock-legacy
- Claude on Google Cloud (rawPredict, `anthropic_version`, Anthropic-managed variant): https://platform.claude.com/docs/en/build-with-claude/claude-models-on-google-cloud/google-managed
- Google ADC order and file paths [2026-10-26]: https://docs.cloud.google.com/docs/authentication/application-default-credentials; metadata server token: https://docs.cloud.google.com/compute/docs/access/authenticate-workloads
- Azure OpenAI v1 auth and `https://ai.azure.com/.default` [2026-05-13]: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle; DefaultAzureCredential chain and production guidance [2026-01-15]: https://learn.microsoft.com/en-us/azure/developer/python/sdk/authentication/credential-chains
- Claude on Foundry endpoint, `api-key` / `x-api-key`, scope `https://cognitiveservices.azure.com/.default`: https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry (via search excerpt, not fetched in full)
- Claude Code refresh races (titles and dates): https://github.com/anthropics/claude-code/issues/54443 (2026-04-28), https://github.com/anthropics/claude-code/issues/61923 (2026-05-24), https://github.com/anthropics/claude-code/issues/50743 (2026-04-19)

Unverified:

- Claude Code `.credentials.json` field names, macOS Keychain service name, refresh endpoint host, JSON body and `client_id` `9d1c250a-e61b-44d9-88ed-5944d1962f5e`: community notes and issue reports only; Claude Code is not open source.
- That subscription tokens need only `oauth-2025-04-20` on `api.anthropic.com`; reported extra server-side checks on requests not coming from Claude Code (2026-01 reports) are not documented.
- Request constraints on `chatgpt.com/backend-api/codex/responses` (required `instructions`, `store`, other headers such as `originator`).
- Gemini CLI encrypted/keychain storage details and env flag name; Antigravity CLI credential store and endpoints.
- Ollama local signing key location (`~/.ollama/id_ed25519`).
- Azure Entra env var names for EnvironmentCredential and WorkloadIdentityCredential (well known, not re-read today).
- SigV4 clock-skew correction behavior per AWS SDK; GCP self-signed JWT without exchange for `aiplatform.googleapis.com`.
- Session review fixes (2026-10-30), after a code review of `session.rs`: write-back goes to the locator the credentials came from, so a `codex` store read from the keychain is written back to the keychain (`NeedWrite {keychain}`; hosts answer it through `security add-generic-password -U`, `secret-tool store` and `CredWriteW`, with a `keychain_write` hook beside `keychain`; a custom reader without a writer answers not-ok). File writes carry `expect`, the `etag` the host reported on the read (lower-case hex SHA-256 of the bytes in every host); the host refuses with `written {ok: false, conflict: true}` when the file changed meanwhile, and on any failed or conflicting write the refreshed token is still used in memory for the call instead of failing it. A 401 re-reads the store for every preset, including `refresh: never` and `claude_code` (the owning tool may have refreshed), and `claude_code_token_env` re-reads the variable; only when refresh is allowed does a refresh follow. An explicit `store` for `codex` never falls back to the keychain of the default home. Token-endpoint errors split three ways: revoked grant (401, 403, `invalid_grant`, `refresh_token_*`) fails `credential_expired`; a rejected request (`invalid_client`, `unauthorized_client`, `unsupported_grant_type`, `invalid_request`, `invalid_scope`) fails `refresh_failed` at once; anything else (5xx, no JSON, transport) keeps a still-valid stored token or, when the token is gone, sleeps 5 s (`NeedSleep`) and sends the token request once more before failing `refresh_failed`. Profile names from `ANTHROPIC_PROFILE` or the config fail `auth{auth_source}` unless they are plain names (no separators, no leading dot), the same rule `active_config` already had. Fixtures: 11 new session cases (write conflict, failed write, keychain write-back, backoff retry, rejected client, 401 re-reads under `never` for `codex`, `claude_code` and the token variable, explicit store without keychain, bad profile name); the fixture format gained an optional `writes` script so a replay can answer a write with a failure.
