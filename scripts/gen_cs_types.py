"""Generates bindings/dotnet/Ducktape.Provider/Types from schema/provider.schema.json."""
import json, pathlib, re, sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
schema = json.load(open(root / "schema/provider.schema.json"))
defs = schema["$defs"]
NAMESPACE = "Ducktape.Types"
OUT = root / "bindings/dotnet/Ducktape.Provider/Types"
OPAQUE = {"raw", "vendor_state", "body", "event", "dialect"}
RENAMES = {"Timeout": "Timeouts"}


def pascal(name):
    return "".join(p[:1].upper() + p[1:] for p in name.split("_") if p)


def const_name(value):
    return pascal(re.sub(r"[^a-z0-9]+", "_", value.lower()))


def is_open_enum(node):
    options = node.get("anyOf") or []
    return len(options) == 2 and "enum" in options[0] and options[1] == {"type": "string"}


def nullable(node):
    options = node.get("anyOf") or []
    return len(options) == 2 and options[1] == {"type": "null"}


extra_types = []
enums = {}
parent_of = {v["$ref"].rsplit("/", 1)[-1]: name for name, node in defs.items() if "oneOf" in node and "properties" not in node for v in node["oneOf"] if "$ref" in v}


def cs_type(node, owner, field):
    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        name = parent_of.get(name, name)
        target = defs[name]
        if is_open_enum(target) or target.get("type") == "string":
            return "string"
        if target.get("type") == "object" and "properties" not in target:
            return cs_type(target, name, "")
        return RENAMES.get(name, name)
    if "const" in node:
        return "string"
    if "enum" in node or is_open_enum(node):
        values = node.get("enum") or node["anyOf"][0]["enum"]
        enums[owner + pascal(field)] = (values, f"{owner}.{field}")
        return "string"
    if nullable(node):
        return cs_type(node["anyOf"][0], owner, field)
    if "anyOf" in node or ("oneOf" in node and "properties" not in node):
        return "object"
    kind = node.get("type")
    if kind == "array":
        return f"List<{cs_type(node.get('items', {}), owner, field)}>"
    if kind == "object":
        if "properties" in node:
            name = owner + pascal(field)
            extra_types.append((name, node))
            return name
        extra = node.get("additionalProperties")
        if isinstance(extra, dict) and extra:
            return f"Dictionary<string, {cs_type(extra, owner, field)}>"
        return "Dictionary<string, object>"
    if field in OPAQUE or kind is None:
        return "JsonElement"
    return {"string": "string", "integer": "long", "number": "double", "boolean": "bool", "null": "object"}[kind]


def doc(text, indent=""):
    text = " ".join((text or "").split()).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return [f"{indent}/// <summary>{text}</summary>"] if text else []


def record(name, node, description=None):
    lines = ["using System.Text.Json;", "using System.Text.Json.Serialization;", "", f"namespace {NAMESPACE};", ""]
    lines += doc(node.get("description") if description is None else description) or [f"/// <summary>One {name} document.</summary>"]
    lines.append(f"public sealed record {name}")
    lines.append("{")
    first = True
    for field, prop in node["properties"].items():
        typ = cs_type(prop, name, field)
        prop_name = pascal(field)
        if prop_name == name:
            prop_name += "Value"
        if not first:
            lines.append("")
        first = False
        lines += doc(prop.get("description"), "    ")
        lines.append(f'    [JsonPropertyName("{field}")]')
        lines.append(f"    public {typ}? {prop_name} {{ get; init; }}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def variant(v):
    return defs[v["$ref"].rsplit("/", 1)[-1]] if "$ref" in v else v


def merged(name, variants, description):
    props = {}
    kinds = sorted({variant(v)["properties"]["type"]["const"] for v in variants})
    for v in variants:
        for field, prop in variant(v)["properties"].items():
            if field == "type":
                props["type"] = {"type": "string", "description": "One of " + ", ".join(kinds) + "; it selects the fields that apply"}
            else:
                props.setdefault(field, prop)
    return record(name, {"properties": props}, description)


def constants(name, values, description):
    lines = [f"namespace {NAMESPACE};", ""]
    lines += doc(description) or [f"/// <summary>Known values of {name}; other strings are allowed.</summary>"]
    lines.append(f"public static class {name}")
    lines.append("{")
    for v in values:
        lines.append(f"    public const string {const_name(v)} = {json.dumps(v)};")
    lines.append("}")
    return "\n".join(lines) + "\n"


files = {}
variant_names = set(parent_of)
for name, node in defs.items():
    if name in variant_names:
        continue
    name = RENAMES.get(name, name)
    if node.get("type") == "object" and "properties" in node:
        files[name] = record(name, node)
    elif "oneOf" in node:
        files[name] = merged(name, node["oneOf"], (node.get("description") or "").strip() + "; the variants are merged into one record")
    elif is_open_enum(node):
        files[name] = constants(name, node["anyOf"][0]["enum"], f"Known values of {name}; other strings are allowed. " + (node.get("description") or ""))
seen = set()
while extra_types:
    name, node = extra_types.pop(0)
    if name in seen:
        continue
    seen.add(name)
    files[name] = record(name, node, node.get("description") or "A nested object")
for type_name, (values, origin) in enums.items():
    if type_name not in files:
        files[type_name] = constants(type_name, values, f"Known values of {origin}; other strings are allowed.")
if "--check" in sys.argv:
    current = {p.name: p.read_text() for p in OUT.glob("*.cs")} if OUT.exists() else {}
    wanted = {f"{n}.cs": t for n, t in files.items()}
    sys.exit(0 if current == wanted else "the C# types are out of date; run python scripts/gen_cs_types.py .")
OUT.mkdir(parents=True, exist_ok=True)
for p in OUT.glob("*.cs"):
    p.unlink()
for n, t in files.items():
    (OUT / f"{n}.cs").write_text(t)
print(f"wrote {len(files)} files to {OUT}")
