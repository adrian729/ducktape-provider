# TODOs

## Implement

- Custom HTTP headers configurable
- Extend image blocks so it is not only base64
- Document content-block type (PDF for example)
- Retry/backoff // "No normalized error taxonomy" -> check both files. Figure out what makes sense and what not to implement. Don't be lazy and just do the less effort choice, actually verify it is the one we should take.
- "There is no shared scenario suite run identically across all three backends" -> this is a true gap, we should fix it. We want to make sure somebody using the provider will be able to use the same generic code, just with different models (only specific config per provider, because even if we want it generic yes, they should be able to tune each provider as they want)

## Investigate

- redacted_thinking, check what this is and if we want to add support to it or not
- Claude pause_turn, wtf is that? Do we want to support it?
- Claude prompt-caching usage fields
- model-capability probing
- Per-request tunables -> check if we do support properly this for each adapter, and if we should add some config options or expose them (specially if they are generic enough)
- "Result shape is thinner than planned" -> check which things from the ones commented are something that really would be worth we add. f.e. it talks about cost estimate, but no idea how are we supposed to do that ourselves... or if we want to put the effort, depends on how easy is to get the information, or if it comes already with the APIs response, etc.
- "No config file / no budgets" -> does this make any sense? Take into account we will just add the provider as a library in another project, that's our goal. No standalone usage expected (apart from testing)
- "Registry: simpler than planned, not necessarily worse" -> what is propossed sounds cool, maybe in parallel with what we already have, not replacing it. But lets first check how hard would it be to implement.
- " `chat`/`stream_chat` as two methods, not one verb with a flag" => lets check this once more, since it was in the original doc. Check what was planned and how, and see if it actually makes sense or what we have si better to make it more clear and be able to distinguish properly return types.
- "Sync-only (open question 3 answered by default, not decided)" => what would this imply? Do we want to support async? Is it worth?
- " **Open question 1 (tools at v0.1 vs v0.2)**: resolved in the more ambitious
  direction — full tool-definition support shipped on all three backends from
  the start, not deferred." => what did it mean by deferred?
- **Streaming event protocol** -> investigate. Differences, benefits, trade-offs, complexity.
