"""Sync baked viewer-demo linear-layout examples from demo_linear_layout.py."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

SYNC_START = "// sync-linear-layout-examples:start"
SYNC_END = "// sync-linear-layout-examples:end"
OUTPUT_AXIS_NAMES = (
    "x",
    "y",
    "z",
    "w",
    "v",
    "u",
    "t",
    "s",
    "r",
    "q",
    "p",
    "o",
    "n",
    "m",
    "l",
    "k",
    "j",
    "i",
    "h",
    "g",
    "f",
    "e",
    "d",
    "c",
    "b",
    "a",
)


def compose_identifier(name: str) -> str:
    """Return one compose-layout-safe identifier."""

    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in name).strip("_")
    if not cleaned:
        return "Layout_1"
    return cleaned if (cleaned[0].isalpha() or cleaned[0] == "_") else f"Layout_{cleaned}"


def viewer_axis_labels(names: list[str]) -> list[str]:
    """Convert dim names into viewer-safe axis labels."""

    counts: dict[str, int] = {}
    labels: list[str] = []
    for name in names:
        base = next((char.upper() for char in name if char.isalpha()), "A")
        index = counts.get(base, 0)
        counts[base] = index + 1
        labels.append(base if index == 0 else f"{base}{index}")
    return labels


def infer_output_dims(input_dims: list[tuple[str, list[list[int]]]]) -> list[tuple[str, int]]:
    """Infer power-of-two output sizes from the highest bit used on each axis."""

    output_names = list(OUTPUT_AXIS_NAMES[: max((len(basis) for _name, bases in input_dims for basis in bases), default=1)])
    sizes = [1] * len(output_names)
    for _dim_name, bases in input_dims:
        for basis in bases:
            for axis, value in enumerate(basis[: len(output_names)]):
                sizes[axis] = max(sizes[axis], 1 if int(value) <= 0 else 1 << int(value).bit_length())
    output_dims = list(zip(output_names, sizes, strict=True))
    return list(reversed(output_dims)) if [name for name, _size in output_dims] == ["x", "y"] else output_dims


def parse_demo_layouts(source_path: Path) -> list[tuple[str, list[tuple[str, list[list[int]]]], str]]:
    """Extract demo titles, bases, and input names from demo_linear_layout.py."""

    module = ast.parse(source_path.read_text())
    demos = next(
        node for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "DEMOS" for target in node.targets)
    )
    if not isinstance(demos.value, ast.Dict):
        raise ValueError("DEMOS must be a dict literal.")
    layouts: list[tuple[str, list[tuple[str, list[list[int]]]], str]] = []
    for entry in demos.value.values:
        if not isinstance(entry, ast.Tuple) or len(entry.elts) != 3:
            raise ValueError("Each DEMOS entry must be a 3-tuple.")
        title = ast.literal_eval(entry.elts[0])
        layout_call = entry.elts[1]
        input_name = ast.literal_eval(entry.elts[2])
        if not isinstance(title, str) or not isinstance(input_name, str):
            raise ValueError("Demo title and input name must be strings.")
        if (
            not isinstance(layout_call, ast.Call)
            or not isinstance(layout_call.func, ast.Attribute)
            or layout_call.func.attr != "from_bases"
            or len(layout_call.args) < 1
        ):
            raise ValueError(f"{title} must call LinearLayout.from_bases(...).")
        input_dims = ast.literal_eval(layout_call.args[0])
        layouts.append((title, input_dims, input_name))
    return layouts


def spec_lines(title: str, input_dims: list[tuple[str, list[list[int]]]]) -> list[str]:
    """Return the compose-layout specs text lines for one demo entry."""

    input_labels = viewer_axis_labels([dim_name for dim_name, _bases in input_dims])
    output_labels = viewer_axis_labels([dim_name for dim_name, _size in infer_output_dims(input_dims)])
    return [
        f"{compose_identifier(title)}: [{','.join(input_labels)}] -> [{','.join(output_labels)}]",
        *[
            f"{axis_label}: {json.dumps(dim_bases, separators=(',', ':'))}"
            for axis_label, (_dim_name, dim_bases) in zip(input_labels, input_dims, strict=True)
        ],
    ]


def const_name(title: str) -> str:
    """Return the generated ts constant name for one demo entry."""

    return f"{compose_identifier(title).upper()}_TEXT"


def format_block(layouts: list[tuple[str, list[tuple[str, list[list[int]]]], str]]) -> str:
    """Return the generated ts block for the baked examples."""

    const_blocks = [
        "\n".join([
            f"const {const_name(title)} = [",
            *[f"    {line!r}," for line in spec_lines(title, input_dims)],
            "].join('\\n');",
        ])
        for title, input_dims, _input_name in layouts
    ]
    example_lines = [
        "const BAKED_EXAMPLES: ExampleState[] = [",
        *[
            f"    bakedExample({title!r}, {const_name(title)}, {compose_identifier(title)!r}, {input_name!r}),"
            for title, _input_dims, input_name in layouts
        ],
        "];",
    ]
    return "\n".join([SYNC_START, *const_blocks, "", *example_lines, SYNC_END])


def main() -> None:
    """Rewrite the baked viewer-demo examples from demo_linear_layout.py."""

    repo_root = Path(__file__).resolve().parents[2]
    source_path = repo_root / "demo_linear_layout.py"
    target_path = repo_root / "tensor-viz" / "packages" / "viewer-demo" / "src" / "linear-layout.ts"
    layouts = parse_demo_layouts(source_path)
    target_text = target_path.read_text()
    pattern = re.compile(rf"{re.escape(SYNC_START)}.*?{re.escape(SYNC_END)}", re.DOTALL)
    replacement = format_block(layouts)
    if not pattern.search(target_text):
        raise ValueError("Missing sync markers in linear-layout.ts.")
    target_path.write_text(pattern.sub(lambda _match: replacement, target_text, count=1))


if __name__ == "__main__":
    main()
