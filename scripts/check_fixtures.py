"""Validates the schemas and checks every fixture under fixtures/types against them."""
import json, sys, pathlib
from jsonschema import Draft202012Validator as V
from referencing import Registry, Resource
root = pathlib.Path(sys.argv[1])
p = json.load(open(root / "schema/provider.schema.json")); e = json.load(open(root / "schema/effects.schema.json"))
V.check_schema(p); V.check_schema(e)
base = "https://github.com/adrian729/ducktape-provider/schema/"
reg = Registry().with_resources([(base + "provider.schema.json", Resource.from_contents(p)), (base + "effects/provider.schema.json", Resource.from_contents(p)), (p["$id"], Resource.from_contents(p)), (e["$id"], Resource.from_contents(e))])
fails = 0
for f in sorted((root / "fixtures/types").glob("*.json")):
    s = json.load(open(f)); d = s["def"]
    sid = e["$id"] if d in e["$defs"] else p["$id"]
    errs = list(V({"$ref": f"{sid}#/$defs/{d}"}, registry=reg).iter_errors(s["value"]))
    if errs: fails += 1; print("FAIL", f.name, [x.message[:100] for x in errs][:2])
for f in sorted((root / "fixtures/wire").glob("*/*.json")):
    s = json.load(open(f))
    docs = [("Start", s["start"])] + [("Step", x["step"]) for x in s["exchange"]] + [("Input", x["input"]) for x in s["exchange"] if x.get("input")]
    for d, v in docs:
        errs = list(V({"$ref": f"{e['$id']}#/$defs/{d}"}, registry=reg).iter_errors(v))
        if errs: fails += 1; print("FAIL", f.parent.name, f.name, d, [x.message[:100] for x in errs][:2]); break
print("fixtures checked, failures:", fails)
sys.exit(1 if fails else 0)
