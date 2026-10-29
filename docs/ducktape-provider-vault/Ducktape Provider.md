# Ducktape Provider

We want a normalized agent/model provider.

- Decide tech stack. Python has the most support probably, but if we can use Cpp or Rust instead it would be nice.
- Check ducktape-laya approach so that we can serve our provider in different languages easily. We don't need to use that if it doesn't fit well or is not the best solution for us, but it would be good to check in case it helps.
- We want to investigate some of the main agent/model provider APIs. We want to support all features possible, but we want normalization on as much as we can, so we want to investigate the APIs, group what can be normalized together between several APIs/providers, what not, etc. The first thing before any implementation would be to define our interface surface already.
- Define the initial API/providers we support. At the very least Claude and GPT/Codex, but it would be good if we check what else is there commonly used and plan for it already so that we dont have surprises and/or big changes later.
- System design and architecture is important. Out interface should account for that. We want a clean interface, meaning deep modules instead of shallow ones, etc. Our defaults should target the most general and common usage, while adding configuration options for when the user wants something more specific.