import assert from "node:assert/strict";
import { test } from "node:test";

import { loadCore } from "../src/core.js";
import { load, wireFixtures } from "./helpers.js";
import { mismatch, replay } from "./replay.js";

const core = await loadCore();

for (const f of wireFixtures()) {
  test(`${f.wire}/${f.name}`, () => {
    const fixture = load(f.path);
    assert.equal(mismatch(replay(core, fixture), fixture.exchange), null);
  });
}
