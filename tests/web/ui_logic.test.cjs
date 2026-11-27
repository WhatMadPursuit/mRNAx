const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.resolve(__dirname, "../../web/app.js"), "utf8");

function app() {
  const fields = new Map();
  const document = {
    addEventListener() {},
    getElementById(id) {
      if (!fields.has(id)) fields.set(id, {});
      return fields.get(id);
    },
  };
  const location = new URL("http://localhost");
  const window = { location, crypto: require("node:crypto").webcrypto };
  const run = (code) => vm.runInContext(code, context);
  const context = vm.createContext({ window, document, URL });
  run(source);
  return { run, document };
}

test("deterministic and export requests omit provider headers", () => {
  const { run } = app();
  run('state.byoKey = "fixture"; state.apiBase = "http://localhost"');
  for (const [method, path] of [["GET", "bundles/id/export"], ["POST", "turn"], ["GET", "agent-chat"]]) {
    const headers = run(`buildHeaders("/v1/projects/p/${path}", "${method}", null, true)`);
    assert.ok(headers["X-MRNAX-Project-Key"]);
    assert.equal(headers["X-MRNAX-LLM-Key"], undefined);
    assert.equal(headers["X-MRNAX-LLM-Provider"], undefined);
  }
  assert.equal(run('buildHeaders("/v1/projects/p/agent-chat", "POST", null, true)["X-MRNAX-LLM-Key"]'), "fixture");
});

test("arbitrary measured axes retain zero, direction, units and explicit missing state", () => {
  const { run } = app();
  run(`const candidate = {metric_details:[
    {metric_id:"orthrus_halflife_pred", value:0, unit:"saluki_pc1_z", direction:"maximize"},
    {metric_id:"extra", value:null, unit:"seconds", direction:"minimize"}
  ]}; const axes = resolveMetricAxes([candidate]);`);
  assert.equal(run("axes.length"), 2);
  assert.match(run("axes[0].label"), /saluki_pc1_z/);
  assert.equal(run("readAxisValue(candidate, axes[0])"), 0);
  assert.equal(run("readAxisValue(candidate, axes[1])"), null);
  assert.match(run("buildSparkCell(null, axes[1], 0, 1)"), /N\/A/);
});

test("older revisions and earlier replies for the same revision cannot replace current state", () => {
  const { run } = app();
  run("state.activeVersion = 3; state.appliedTurnSequence = 5");
  assert.equal(run('applyTurnResult({intent:{intent_version:2}}, "", 7)'), false);
  assert.equal(run('applyTurnResult({intent:{intent_version:3}}, "", 4)'), false);
  assert.equal(run("state.activeVersion"), 3);
  assert.equal(run("state.appliedTurnSequence"), 5);
});
