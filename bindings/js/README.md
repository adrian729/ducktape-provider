# ducktape-provider

One normalized interface over model vendors: Anthropic, OpenAI, Google and Ollama behind one `Provider`, on Node, Electron and the browser.

```js
import { Provider, user } from "ducktape-provider";

const p = new Provider("anthropic", { default_model: "claude-opus-5-5" });
console.log((await p.chat([user("hello")])).text);
```

Documentation: https://github.com/adrian729/ducktape-provider
