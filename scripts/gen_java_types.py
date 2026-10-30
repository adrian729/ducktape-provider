"""Generates bindings/java/src/main/java/io/github/adrian729/ducktape/types from schema/provider.schema.json."""
import json, pathlib, re, sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
schema = json.load(open(root / "schema/provider.schema.json"))
defs = schema["$defs"]
PACKAGE = "io.github.adrian729.ducktape.types"
OUT = root / "bindings/java/src/main/java" / PACKAGE.replace(".", "/")
OPAQUE = {"raw", "vendor_state", "body", "event", "dialect"}


def pascal(name):
    return "".join(p[:1].upper() + p[1:] for p in name.split("_") if p)


def lower(name):
    p = pascal(name)
    return p[:1].lower() + p[1:]


def const_name(value):
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def is_open_enum(node):
    options = node.get("anyOf") or []
    return len(options) == 2 and "enum" in options[0] and options[1] == {"type": "string"}


def nullable(node):
    options = node.get("anyOf") or []
    return len(options) == 2 and options[1] == {"type": "null"}


extra_types = []
enums = {}
parent_of = {v["$ref"].rsplit("/", 1)[-1]: name for name, node in defs.items() if "oneOf" in node and "properties" not in node for v in node["oneOf"] if "$ref" in v}


def java_type(node, owner, field):
    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        name = parent_of.get(name, name)
        target = defs[name]
        if is_open_enum(target) or target.get("type") == "string":
            return "String"
        if target.get("type") == "object" and "properties" not in target:
            return java_type(target, name, "")
        return name
    if "const" in node:
        return "String"
    if "enum" in node or is_open_enum(node):
        values = node.get("enum") or node["anyOf"][0]["enum"]
        enums[owner + pascal(field)] = (values, f"{owner}.{field}")
        return "String"
    if nullable(node):
        return java_type(node["anyOf"][0], owner, field)
    if "anyOf" in node or ("oneOf" in node and "properties" not in node):
        return "Object"
    kind = node.get("type")
    if kind == "array":
        return f"List<{java_type(node.get('items', {}), owner, field)}>"
    if kind == "object":
        if "properties" in node:
            name = owner + pascal(field)
            extra_types.append((name, node))
            return name
        extra = node.get("additionalProperties")
        if isinstance(extra, dict) and extra:
            return f"Map<String, {java_type(extra, owner, field)}>"
        return "Map<String, Object>"
    if field in OPAQUE or kind is None:
        return "JsonNode"
    return {"string": "String", "integer": "Long", "number": "Double", "boolean": "Boolean", "null": "Object"}[kind]


def doc(text, indent=""):
    text = " ".join((text or "").split())
    return [f"{indent}/** {text} */"] if text else []


def record(name, node, description=None):
    fields = []
    for field, prop in node["properties"].items():
        fields.append((field, java_type(prop, name, field), prop.get("description")))
    lines = [f"package {PACKAGE};", "", "import com.fasterxml.jackson.annotation.JsonProperty;", "import com.fasterxml.jackson.databind.JsonNode;", "import java.util.List;", "import java.util.Map;", ""]
    lines += doc(node.get("description") if description is None else description) or [f"/** One {name} document. */"]
    params = []
    for field, typ, d in fields:
        params.append(f"    @JsonProperty(\"{field}\") {typ} {lower(field)}")
    lines.append(f"public record {name}(")
    lines.append(",\n".join(params))
    lines.append(") {")
    lines.append(f"    /** A builder with every field unset. */")
    lines.append(f"    public static Builder builder() {{ return new Builder(); }}")
    lines.append("")
    lines.append(f"    /** Builds a {name} field by field. */")
    lines.append(f"    public static final class Builder {{")
    for field, typ, d in fields:
        lines.append(f"        private {typ} {lower(field)};")
    for field, typ, d in fields:
        lines += doc(d, "        ")
        lines.append(f"        public Builder {lower(field)}({typ} value) {{ this.{lower(field)} = value; return this; }}")
    lines.append(f"        public {name} build() {{ return new {name}({', '.join(lower(f) for f, _, _ in fields)}); }}")
    lines.append("    }")
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
    lines = [f"package {PACKAGE};", ""]
    lines += doc(description) or [f"/** Known values of {name}; other strings are allowed. */"]
    lines.append(f"public final class {name} {{")
    lines.append(f"    private {name}() {{}}")
    for v in values:
        lines.append(f"    public static final String {const_name(v)} = {json.dumps(v)};")
    lines.append("}")
    return "\n".join(lines) + "\n"


files = {}
variant_names = set(parent_of)
for name, node in defs.items():
    if name in variant_names:
        continue
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
    current = {p.name: p.read_text() for p in OUT.glob("*.java")} if OUT.exists() else {}
    wanted = {f"{n}.java": t for n, t in files.items()}
    sys.exit(0 if current == wanted else "the Java types are out of date; run python scripts/gen_java_types.py .")
OUT.mkdir(parents=True, exist_ok=True)
for p in OUT.glob("*.java"):
    p.unlink()
for n, t in files.items():
    (OUT / f"{n}.java").write_text(t)
print(f"wrote {len(files)} files to {OUT}")
