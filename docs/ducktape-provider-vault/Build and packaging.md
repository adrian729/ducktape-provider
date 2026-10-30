# Build and packaging

How to build, package and release one Rust sans-I/O core for eight targets. As of 2026-10-28. Context: [[Design draft]] (Architecture, Language rollout, decision 2) and [[Delivery across languages]].

## Recommendation

Two compilation targets of the same core, one ABI:

- **One `core.wasm`** (wasm32-unknown-unknown, plain `extern "C"` exports, zero imports) serves JS (Node, Electron, browser, Deno, Bun), Go (wazero) and Java/Kotlin (Chicory). Built once on one Linux job. No platform matrix for these three ecosystems. Java also ships a second artifact, `ducktape-provider-native`, FFM over the native library (JDK 22+), same API, for users who want the 11 µs path; decided 2026-10-28.
- **One native C ABI** (same `extern "C"` source, `cdylib` + `staticlib`, cbindgen header) serves C/C++ and .NET (P/Invoke, one NuGet with `runtimes/{rid}/native`).
- **PyO3 + maturin abi3 wheels** for Python (native objects, no JSON hop). One wheel per platform, plus `cp315-abi3.abi3t` for 3.15+ free-threaded.
- **Source crate** on crates.io for Rust.
- CI: 4 native runners (Linux x64, Linux arm64, Windows x64, macOS arm64 also building macOS x64) + 1 wasm job. Build where you test; no cross toolchains, no Docker.
- Release: tag `vX.Y.Z` drives one workflow; single version in `[workspace.package]`; trusted publishing on PyPI, npm, crates.io, NuGet; Maven Central by token + GPG; Go from `bindings/go` in the monorepo (decision 13), tagged `bindings/go/vX.Y.Z` with `core.wasm` committed there at the release commit.
- No cargo-dist, no napi-rs, no wasm-pack, no cross-rs.

Rationale: wasm removes the matrix exactly where packaging native code is painful (Go modules are VCS-only; Java needs JDK 22+ for FFM and extraction tricks; npm needs one package per platform). Where wasm would not remove the matrix (Python, .NET, C need a native runtime anyway), native is smaller and faster. Measured wasm cost is 2-12x native per event, in microseconds, against tens of ms between tokens ([[Delivery across languages]], Performance reality).

Accepted as decision 14 in [[Design draft]] (Architecture, Language rollout): Go and Java on the wasm build; .NET and C on the C ABI; Node and browser on the same plain wasm, not wasm-bindgen.

## 1. Boundary per language

Options: (a) native cdylib + C ABI, (b) language-native binding, (c) one wasm everywhere.

| Target | (a) cdylib + C ABI | (b) native binding | (c) one wasm | Pick |
|---|---|---|---|---|
| Python | ctypes/cffi; wheels per platform anyway; ctypes + json.loads 48 µs/event measured | PyO3 0.30 + maturin 1.15: 1 abi3 wheel per platform, + abi3.abi3t for 3.15+; Python objects directly | wasmtime-py is itself a native wheel; no gain | (b) |
| Node / Electron | node-addon-api or napi-rs prebuilds per platform | napi-rs 3.15: one npm package per platform via `optionalDependencies`; multi-package publish not atomic | one npm package, no matrix; 13 µs/event measured; Electron main and renderer without rebuild | (c) |
| Browser | n/a | wasm-bindgen 0.2.129: typed glue, separate outputs per target (`web`, `nodejs`, `bundler`, `module`) | same `core.wasm` as Node | (c) |
| Go | purego 0.11 (no cgo) or cgo; per-platform libs committed and extracted at runtime (Turso) | n/a | wazero 1.12: pure Go, `CGO_ENABLED=0`, one embedded file (1Password); 17 µs/event; ~0.5 s compile at init | (c) |
| Java / Kotlin | FFM final only in JDK 22+, jextract; natives in jar + extraction; native-access warning since JDK 24; 11 µs/event | JNI: worse than FFM | Chicory 1.7.5: pure Java 11+, Android-tested, build-time compiler to `.class`; 60 µs/event runtime-compiled | (c) |
| .NET | P/Invoke (`LibraryImport`) or csbindgen 1.9.8; one NuGet with `runtimes/{rid}/native` | n/a | Wasmtime NuGet 48.0.2 is 53 MB with 17-31 MB native libs per RID; WACS 0.9 is pure C# but solo-maintained | (a) |
| C / C++ | cbindgen 0.29.4 header + `.so`/`.dll`/`.dylib` + static lib per platform | = (a) | would need a wasm runtime in C | (a) |
| Rust | n/a | source crate, no matrix | n/a | (b) |

Where wasm removes the platform matrix:

| Ecosystem | Matrix removed by wasm | Why |
|---|---|---|
| JS (Node, Electron, browser, Deno, Bun) | yes | V8/JSC/SpiderMonkey run wasm natively |
| Go | yes | wazero is pure Go; `go:embed` the file |
| Java / Kotlin | yes | Chicory is pure Java; build-time compiler emits bytecode |
| .NET | no | maintained runtimes (Wasmtime, Extism) ship native libs per RID |
| Python | no | runtimes (wasmtime-py) are native wheels |
| C / C++ | no | host needs a native library either way |
| Rust | n/a | source distribution |

Precedents:

- 1Password Go SDK: Extism go-sdk on wazero, `internal/wasm/core.wasm` (9.5 MB) committed in a separate Go repo, plugin guarded by a `sync.Mutex`; a cgo shared-lib path exists only for desktop-app auth.
- Turso Go: purego, no cgo; native libs in a separate module `turso-go-platform-libs`, one `go:embed` file per GOOS/GOARCH, `purego.Dlopen` after extraction; musl variants are static `.a`.
- Extism: Go SDK on wazero; .NET SDK needs `Extism.runtime.all` (6 native RID packages). Confirms .NET wasm is not pure.

Runtime cost of (c), measured 2026-10-28 on the probe crate (section 5), SSE stream of 2001 text deltas fed in 97-byte chunks. Machine load average ~9 on 8 cores: relative numbers only.

| Path | µs per event | Startup |
|---|---|---|
| Rust native, in-process | 5.1 | - |
| Java 27 FFM to cdylib (no host JSON) | 11.1 | 361 ms (lookup + first link) |
| Node 26 + `core.wasm` O3 (incl. JSON.parse) | 13.2 | 21 ms instantiate |
| Node 26 + `core.wasm` Oz (incl. JSON.parse) | 20.8 | 4 ms instantiate |
| Go 1.27 + wazero compiler (no host JSON) | 16.6 | 0.53-0.59 s compile, 0.2-0.3 ms instantiate |
| Go 1.27 + wazero compiler + `encoding/json` to map | 30.4 | same |
| Python 3.15 ctypes to cdylib + json.loads | 47.8 | - |
| Java 27 + Chicory runtime compiler (no host JSON) | 60.1 | 4.0 s |
| Go wazero interpreter | 1231 | 0.1 s |
| Java Chicory interpreter (default) | 3228 | 1.5 s |

Implications:

- Never ship interpreter mode. Chicory: use the build-time compiler (Maven plugin; output shipped as classes in our jar, so Gradle users are unaffected). wazero: compiler mode is the default on amd64/arm64; compile once per process, share the `CompiledModule`, optionally `NewCompilationCacheWithDir`.
- JSON through linear memory = alloc + copy in + call + copy out + free per call. Cost is in the table; it is noise against inter-token gaps.
- Instances are single-threaded and own their memory. Core state (operation, decoder) lives in one instance, so pin an operation to an instance for its lifetime: instance per operation (wazero instantiate 0.2-0.3 ms) or a small pool. No wasm threads needed: sans-I/O state is per operation.
- ABI rules that keep (a) and (c) identical: only pointers, `size_t`/`u32` lengths and opaque handles; no structs by value, no callbacks; `alloc`/`free` exported; `abi_version()` export; JSON UTF-8 in and out.

## 2. Workspace layout

```
Cargo.toml              workspace; [workspace.package] version; profiles
crates/
  core/                 ducktape-core: pure lib, serde + serde_json only, #![forbid(unsafe_code)]  -> crates.io
  abi/                  ducktape-abi: extern "C" surface; crate-type cdylib + staticlib; cbindgen header
                        native target -> C/C++ and .NET; wasm32-unknown-unknown target -> core.wasm
  py/                   ducktape-py: PyO3 module (maturin); not in default-members
  provider/             (phase 2) ducktape-provider: Rust API + optional reqwest feature -> crates.io
  conformance/          fixture runner for core, abi (JSON) and the PyO3 objects; publish = false
bindings/
  js/  go/  java/  dotnet/  c/  python/     thin host glue + per-language conformance runners
schema/                 JSON Schema of the normalized types (the contract)
fixtures/               JSON fixtures validated against it
```

- `default-members` = core, abi, conformance: `cargo test` needs no Python, Node, Go, JDK or .NET.
- `unsafe` only in `abi`. `core` has no `extern`, no I/O, no global state.
- Dependency policy for `core`: `serde` (derive) + `serde_json` (`raw_value` to pass vendor JSON through without re-parsing); for auth: `sha2`, `hmac`, `rsa` (RustCrypto, build for wasm32-unknown-unknown), `zeroize`, `base64`. No `thiserror`, `regex`, `chrono`, `indexmap`, async, logging crates, no `ring`. Proc macros: `serde_derive` only. No randomness in core: the host passes entropy.
- Not simd-json: needs mutable input buffers, lots of `unsafe`, SIMD-only payoff on large documents; events are small.
- Measured probe build (Rust 1.98.0, loaded machine): 14 crates in the graph; clean debug 16-17 s, clean `cargo check` 7.6 s, clean release (fat LTO) 28 s, clean wasm release 24 s, incremental debug after touching the core 0.9 s. Per-unit times: serde_core 6.1 s, serde_derive 5.0 s, syn 4.4 s, serde_json 3.0 s. The syn -> serde_derive chain is the critical path and is paid once per cache.

## 3. Cross-compilation and CI

| Tool | Status 2026-10 | Role here |
|---|---|---|
| GitHub arm64 Linux runners `ubuntu-24.04-arm` | GA for public repos 2025-08; private repos 2026-01 | native Linux arm64 build + test |
| `macos-15` (arm64) | current | builds aarch64 and x86_64 Apple targets |
| `macos-15-intel` | last x86_64 image | optional x86_64 test job until it retires |
| cargo-zigbuild 0.24.0 | active | not needed; fallback for a glibc floor (`--target x86_64-unknown-linux-gnu.2.17`) or Linux-to-macOS builds; no Windows |
| cargo-xwin 0.23.1 | active | not needed; Windows MSVC from Linux |
| cross 0.2.5 | last release 2023-02 | skip (Docker; napi-rs calls it legacy) |
| maturin 1.15.0 + maturin-action v1.51.0 | active | wheels; manylinux2014 (glibc 2.17) container or `--zig`; abi3; abi3t since 1.14 |
| napi-rs 3.15 | active | not needed (wasm) |
| wasm-bindgen 0.2.129 | moved to `wasm-bindgen` org after rustwasm sunset (2025-07) | not needed (plain exports) |
| wasm-pack 0.15.0 | maintained by drager under `wasm-bindgen` org | not needed |
| binaryen `wasm-opt` version_133 | active | one `wasm-opt -O3` step on `core.wasm` |
| Swatinem/rust-cache v2.9.3 | active | caches registry + dependency artifacts; enough for 14 deps |
| sccache 0.19.0 (GHA backend) | active | not needed at this size |

Linux glibc floor: a cdylib built on `ubuntu-24.04` may require that runner's newer glibc. Build the C/.NET Linux libs inside the manylinux2014 container that maturin-action already uses, or with `cargo zigbuild --target <triple>.2.17`. Wheels get the floor from maturin.

Matrix (5 jobs; PRs can run only the Linux x64 and wasm jobs):

| Job | Runner | Builds | Tests |
|---|---|---|---|
| linux-x64 | `ubuntu-24.04` | cdylib + staticlib (glibc 2.17), abi3 + abi3t wheels, sdist, header | core + conformance, Python, .NET, C smoke |
| linux-arm64 | `ubuntu-24.04-arm` | same | same |
| windows-x64 | `windows-2025` | `.dll` + import `.lib` + static `.lib`, wheels | same |
| macos | `macos-15` | aarch64 + x86_64 dylib/static, wheels for both | arm64 native; x86_64 only if Rosetta or `macos-15-intel` |
| wasm | `ubuntu-24.04` | `core.wasm` + wasm-opt, npm tarball, Java jar (Chicory build-time compile), Go tree | Node, headless browser, Go, Java conformance against the same file |

Single-runner alternative: zig + xwin on one Linux box can build every native target for a pure-Rust crate, but the conformance suite still needs real runners, so it saves nothing.

### Status (2026-10-30)

The build matrix above is wired as `.github/workflows/package.yml` for wheels (abi3, five targets), the sdist and the npm tarball with `wasm-opt`; artifacts only, no publishing yet. The Go tree builds and tests in the CI `go` job against a freshly built `core.wasm`; the file is still gitignored and gets committed only at release commits (decision 13). The C, .NET and Java artifacts are not built yet.

## 4. Release and distribution

| Target | Artifact | Registry | Auth | Provenance |
|---|---|---|---|---|
| Python | abi3 wheels (from 3.10) + `cp315-abi3.abi3t` per platform + sdist | PyPI | trusted publishing (OIDC) | PEP 740 attestations by default with `pypa/gh-action-pypi-publish`; `uv publish` does not create them |
| JS / Electron / browser | one package: `core.wasm` + ESM wrapper + `.d.ts` from JSON Schema | npm | trusted publishing, GA 2025-07-31, npm CLI 11.5.1+ | provenance automatic |
| Go | `bindings/go` sources + committed `core.wasm`, tag `bindings/go/vX.Y.Z` | proxy.golang.org / sum.golang.org | none (git tag) | checksum DB; `actions/attest` on the wasm |
| Java / Kotlin | `ducktape-provider`: wrapper + Chicory-compiled classes, `chicory-runtime` dependency; `ducktape-provider-native`: wrapper + FFM loader + natives for 5 platforms | Maven Central (Central Portal) | Portal user token + GPG key; no OIDC found | PGP signatures required; Sigstore bundles optional |
| .NET | one nupkg, `runtimes/{linux-x64,linux-arm64,win-x64,osx-arm64,osx-x64}/native` | nuget.org | trusted publishing (2025-09-22) via `NuGet/login` | `actions/attest` on the nupkg |
| Rust | `ducktape-core` (+ `ducktape-provider` later) | crates.io | trusted publishing (2025-07-11); first publish manual | none native; optional `actions/attest` on `.crate` |
| C / C++ | per-platform archive: header, shared + static lib, pkg-config / CMake config | GitHub Releases | `GITHUB_TOKEN` | `actions/attest`: SLSA Build L2, L3 via reusable workflow |

- GitHub artifact attestations: free for public repos; private repos need Enterprise Cloud. `actions/attest-build-provenance` v4 is a wrapper over `actions/attest`.
- Maven Central: OSSRH shut down 2025-06-30. Publishing limits enforced since 2026-10-01 target commercial volume; community open source stays free.
- cargo-dist 0.33.1 (2026-10-21): maintained, but built for CLI binaries and installers; library archives (`package-libraries`) are experimental since 0.20; no wheels, NuGet, Maven or Go. Not worth adopting.
- How consumers get the binary with no build step:
  - Go: module zips come from VCS, so the binary must be committed at the tagged commit. wasm = one ~250 KB file committed under `bindings/go` at each release commit (about 90 KB packed per release in history). Native would mean committing 5 libs per release and extracting at runtime (Turso pattern).
  - Java: classes in the jar (build-time compiled) or `core.wasm` as a classpath resource. The FFM alternative needs natives in the jar, temp-dir extraction, JDK 22+, and `--enable-native-access=ALL-UNNAMED` to silence the JDK 24+ warning (the `Enable-Native-Access` manifest attribute works only for `java -jar` executable jars).
  - .NET: NuGet resolves `runtimes/{rid}/native` automatically, no custom resolver.
  - JS: `new URL('./core.wasm', import.meta.url)` + `WebAssembly.instantiateStreaming` in browsers, `fs` in Node; bundlers handle the URL pattern.

## 5. Binary size and startup

Probe: ~150 lines of tagged-enum types, a request builder, an SSE decoder and a C ABI, serde + serde_json only. Linux x64, Rust 1.98.0, measured 2026-10-28.

| Artifact | Profile | Raw bytes | gzip -9 |
|---|---|---|---|
| cdylib `.so` | release defaults (no LTO, cgu 16, unwind), unstripped | 4,872,408 | - |
| cdylib `.so` | same, `strip = true` | 610,416 | - |
| cdylib `.so` | O3, fat LTO, cgu 1, abort, strip | 571,400 | 272,639 |
| cdylib `.so` | same with `opt-level = "z"` | 535,696 | 244,757 |
| `core.wasm` | O3, fat LTO, cgu 1, abort, strip | 252,928 | 91,363 |
| `core.wasm` | same with `opt-level = "z"` | 209,158 | 71,109 |

- `strip` is the only large win (8x). LTO, cgu 1 and abort add ~6%. `opt-level = "z"`: -6% native, -17% wasm, but Node per-event time went from 13.2 to 20.8 µs. Keep O3.
- `wasm-opt`: typically another 15-20% (rustwasm book); not measured here.
- `core.wasm` has zero imports: no WASI, no JS glue required.
- `panic`: keep `unwind` for native. PyO3 turns panics into `PanicException` only when unwinding, and the C ABI should `catch_unwind` and return an error, because since Rust 1.81 a panic escaping `extern "C"` aborts the host process. Use `abort` for wasm (a panic traps either way).
- Estimate for the real core (4 wires): 2-3x the probe's code, so native well under 1 MB and wasm under 500 KB raw. Meets the [[Design draft]] targets (Python wheel and wasm under 1 MB).

```toml
[profile.release]
lto = "fat"
codegen-units = 1
strip = true

[profile.release-wasm]
inherits = "release"
panic = "abort"
```

## 6. Versioning

- One source: `[workspace.package] version`, every crate `version.workspace = true`. maturin reads it. npm, Maven and .NET versions are stamped from the tag at build (`npm version --no-git-tag-version`, `-Drevision=`, `-p:Version=`), so no other manifest stores it. CI fails if the tag differs from the workspace version.
- Flow: bump the workspace version, commit `core.wasm` under `bindings/go`, push tags `vX.Y.Z` and `bindings/go/vX.Y.Z` on the same commit. One workflow runs the 5 jobs, then publishes PyPI, npm, crates.io, NuGet, Maven Central, GitHub release archives. Publishing is not atomic across registries; never republish different bits under a version, re-run only the missing steps.
- release-please (v17.11.2) is optional for release PRs and changelog: one root `rust` component, `include-component-in-tag: false` gives plain `vX.Y.Z`. Its release types include rust, node, python, maven, go and simple, but not .NET.
- release-plz (0.4.2) covers Rust crates only.
- Go tags (decision 13: monorepo): the module in `bindings/go` is tagged `bindings/go/vX.Y.Z`, and `core.wasm` is committed there at that commit. The rejected alternative was a separate Go repo with plain tags and no binaries in the main history; the price of staying in one repo is the second tag and about 90 KB of packed history per release. At v2+ the module path needs a `/v2` suffix.

## Risks

- Rust may enable new wasm features by default for `wasm32-unknown-unknown`; CI must run the same `core.wasm` on wazero and Chicory. Rust 1.98 output ran on both.
- Startup: wazero compile ~0.5 s for 250 KB (use a compilation cache); Chicory runtime compile 4 s (use the build-time compiler).
- Maven Central needs long-lived secrets (token + GPG).
- Free-threaded Python 3.14t has no stable ABI: ship a `cp314t` wheel or let it build from sdist.
- macOS x86_64 testing after `macos-15-intel` retires.

## Verified / Unverified

Verified (read 2026-10-28; date = publication or release):

- Arm64 runners GA in public repos, 2025-08-07: https://github.blog/changelog/2025-08-07-arm64-hosted-runners-for-public-repositories-are-now-generally-available/
- Arm64 runners in private repos, 2026-01-29 (official changelog, seen via search result): https://github.blog/changelog/2026-01-29-arm64-standard-runners-are-now-available-in-private-repositories
- crates.io trusted publishing, 2025-07-11; GitLab support, enforcement, blocked triggers 2026-01-21: https://blog.rust-lang.org/2025/07/11/crates-io-development-update-2025-07 , https://blog.rust-lang.org/2026/01/21/crates-io-development-update
- npm trusted publishing GA, 2025-07-31: https://github.blog/changelog/2025-07-31-npm-trusted-publishing-with-oidc-is-generally-available/ ; https://docs.npmjs.com/trusted-publishers
- NuGet trusted publishing, 2025-09-22: https://devblogs.microsoft.com/dotnet/enhanced-security-is-here-with-the-new-trust-publishing-on-nuget-org/ ; https://learn.microsoft.com/nuget/nuget-org/trusted-publishing
- PyPI attestations (PEP 740, default in the pypa action): https://docs.pypi.org/attestations/ (2024-11 launch)
- OSSRH end of life 2025-06-30: https://central.sonatype.org/news/20250326_ossrh_sunset/
- Maven Central publishing limits, 2026-06-16, updated 2026-10-01: https://www.sonatype.com/blog/open-publishing-commercial-scale
- GitHub artifact attestations, SLSA Build L2/L3, plan availability: https://docs.github.com/actions/security-guides/using-artifact-attestations-to-establish-provenance-for-builds ; https://github.com/actions/attest-build-provenance
- cargo-dist v0.33.1, 2026-10-21; `package-libraries` experimental since 0.20: https://github.com/axodotdev/cargo-dist , https://github.com/axodotdev/cargo-dist/blob/main/book/src/reference/config.md
- rustwasm org sunset, 2025-07-21: https://blog.rust-lang.org/inside-rust/2025/07/21/sunsetting-the-rustwasm-github-org
- wasm-pack v0.15.0, 2026-05-15, maintained by drager: https://github.com/wasm-bindgen/wasm-pack
- wasm-bindgen 0.2.129, 2026-09-25; targets: https://github.com/wasm-bindgen/wasm-bindgen , https://wasm-bindgen.github.io/wasm-bindgen/reference/deployment.html
- maturin v1.15.0, 2026-08-24; distribution (manylinux, zig, xwin, generate-ci): https://github.com/PyO3/maturin , https://www.maturin.rs/distribution.html
- PyO3 v0.30.1, 2026-10-14; abi3 / abi3t, maturin >= 1.14: https://pyo3.rs/main/building-and-distribution.html
- PEP 803 abi3t, Final, resolved 2026-03-30: https://peps.python.org/pep-0803/
- napi-rs 3.15.0, 2026-10-22; cross-build and release docs: https://napi.rs/docs/cross-build , https://napi.rs/docs/deep-dive/release
- cargo-zigbuild v0.24.0 2026-10-21; cargo-xwin v0.23.1 2026-08-13; cross v0.2.5 2023-02-04: https://github.com/rust-cross/cargo-zigbuild , https://github.com/rust-cross/cargo-xwin , https://github.com/cross-rs/cross
- sccache v0.19.0 2026-10-24 (GHA backend): https://github.com/mozilla/sccache/blob/main/docs/GHA.md ; rust-cache v2.9.3 2026-10-20: https://github.com/Swatinem/rust-cache
- binaryen version_133, 2026-09-21: https://github.com/WebAssembly/binaryen
- wazero v1.12.0, 2026-05-29 (repo now `wazero/wazero`; `go get github.com/tetratelabs/wazero@v1.12.0` resolves): https://github.com/wazero/wazero , https://pkg.go.dev/github.com/tetratelabs/wazero
- Chicory 1.7.5, 2026-03-24; `maven.compiler.release` 11; build-time compiler: https://github.com/dylibso/chicory , https://chicory.dev/docs/usage/build-time-compiler
- Wasmtime NuGet 48.0.2, 2026-09-18, 53 MB with native libs for 6 RIDs (package listed locally): https://www.nuget.org/packages/Wasmtime
- WACS 0.9.0, 2026-04-23, solo maintainer: https://github.com/kelnishi/WACS
- Extism 1.41.0, 2026-10-09; .NET needs native runtime packages: https://github.com/extism/extism , https://www.nuget.org/packages/Extism.runtime.all
- 1Password Go SDK: `go.mod` (extism go-sdk v1.7.1, wazero), `internal/wasm/core.wasm`, mutex: https://github.com/1Password/onepassword-sdk-go
- Turso Go via purego + embedded platform libs: https://github.com/tursodatabase/turso/tree/main/bindings/go , https://github.com/tursodatabase/turso-go-platform-libs
- purego v0.11.2, 2026-10-18, platform tiers: https://github.com/ebitengine/purego
- csbindgen 1.9.8, 2026-05-21: https://github.com/Cysharp/csbindgen ; cbindgen 0.29.4, 2026-06-10: https://github.com/mozilla/cbindgen ; jextract: https://github.com/openjdk/jextract
- JEP 472 (JDK 24): native access warns by default; manifest attribute only for executable jars: https://openjdk.org/jeps/472 . Warning reproduced on JDK 27.
- Go modules: subdirectory tag prefix, zip limits (500 MiB), nested modules excluded: https://go.dev/ref/mod
- release-please v17.11.2, 2026-08-24; release types, config schema (`tag-separator`, `include-component-in-tag`, `extra-files`): https://github.com/googleapis/release-please ; release-plz 0.4.2, 2026-10-27: https://release-plz.dev/docs
- Rust 1.81: panic escaping `extern "C"` aborts: https://blog.rust-lang.org/2024/09/05/Rust-1.81.0/
- wasm-opt 15-20%: https://rustwasm.github.io/book/reference/code-size.html ; size profile: https://github.com/johnthagen/min-sized-rust
- simd-json trade-offs: https://github.com/simd-lite/simd-json
- Electron native modules need rebuilding: https://www.electronjs.org/docs/latest/tutorial/using-native-node-modules
- Local measurements, 2026-10-28: Rust 1.98.0, Node 26.7.0, Go 1.27.2, OpenJDK 27, Python 3.15.0, wazero 1.12.0, Chicory 1.7.5. Load average ~9 on 8 cores.

Unverified:

- Maven Central has no OIDC trusted publishing: only third-party PRs/issues (Sep-Oct 2026); Sigstore bundles validated but optional since 2025-01 (secondary source).
- `macos-15-intel` support until ~August 2027 (secondary sources disagree on the month).
- Rosetta 2 on GitHub `macos-15` arm64 images (not in the sources found).
- 2 vCPU for arm64 runners in private repos (one secondary source).
- wasm-pack runs wasm-opt by default in release (mirror docs only); irrelevant to the recommendation.
- `wasm-opt` gain on our binary: running a downloaded binaryen was blocked in this session.
- Chicory build-time compiler speed: not measured (needs the Maven plugin); expected near the runtime-compiler steady state without the 4 s init.
- wazero `api.Module` goroutine safety: docs do not say; treat as not safe (1Password uses a mutex).
- Whether release-please's `rust` strategy bumps `[workspace.package] version`.
- Clean build times on an idle machine: all measured under load.
- Node-API addons loading in Electron without rebuild (prebuildify README says Node-API builds work on Electron > 3); moot with wasm.
