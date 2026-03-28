#!/usr/bin/env python3
"""Convert ONNX model to plain HTML for viewing in w3m."""

import argparse
import html
import sys
from typing import Dict, List

import onnx
from onnx import (AttributeProto, ModelProto, NodeProto, TensorProto,
                  ValueInfoProto, shape_inference)


def escape(text: str) -> str:
    """HTML escape text."""
    return html.escape(str(text))


def format_shape(value_info: ValueInfoProto) -> str:
    """Format tensor shape as string."""
    if not value_info.type.HasField("tensor_type"):
        return "unknown"

    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return "unknown"

    dims = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(str(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")

    return f"[{', '.join(dims)}]"


def get_tensor_type_name(dtype: int) -> str:
    """Get tensor data type name from ONNX type enum."""
    type_map: Dict[int, str] = {
        TensorProto.FLOAT: "float32",
        TensorProto.UINT8: "uint8",
        TensorProto.INT8: "int8",
        TensorProto.UINT16: "uint16",
        TensorProto.INT16: "int16",
        TensorProto.INT32: "int32",
        TensorProto.INT64: "int64",
        TensorProto.STRING: "string",
        TensorProto.BOOL: "bool",
        TensorProto.FLOAT16: "float16",
        TensorProto.DOUBLE: "float64",
        TensorProto.UINT32: "uint32",
        TensorProto.UINT64: "uint64",
        TensorProto.COMPLEX64: "complex64",
        TensorProto.COMPLEX128: "complex128",
        TensorProto.BFLOAT16: "bfloat16",
    }
    result = type_map.get(dtype)
    if result is not None:
        return result
    return f"unknown({dtype})"


def format_value_info(value_info: ValueInfoProto) -> str:
    """Format ValueInfo as string with type and shape."""
    if not value_info.type.HasField("tensor_type"):
        return f"{escape(value_info.name)}: unknown"

    tensor_type = value_info.type.tensor_type
    dtype_name = get_tensor_type_name(tensor_type.elem_type)
    shape = format_shape(value_info)

    return f"{escape(value_info.name)}: {dtype_name}{shape}"


def format_attribute(attr: AttributeProto) -> str:
    """Format node attribute as string."""
    attr_type = attr.type

    if attr_type == onnx.AttributeProto.FLOAT:
        return str(attr.f)
    elif attr_type == onnx.AttributeProto.INT:
        return str(attr.i)
    elif attr_type == onnx.AttributeProto.STRING:
        result: str = attr.s.decode("utf-8", errors="replace")
        return result
    elif attr_type == onnx.AttributeProto.TENSOR:
        return "<tensor>"
    elif attr_type == onnx.AttributeProto.GRAPH:
        return "<graph>"
    elif attr_type == onnx.AttributeProto.FLOATS:
        return str(list(attr.floats))
    elif attr_type == onnx.AttributeProto.INTS:
        return str(list(attr.ints))
    elif attr_type == onnx.AttributeProto.STRINGS:
        return str([s.decode("utf-8", errors="replace") for s in attr.strings])
    else:
        return f"<type:{attr_type}>"


def generate_html(model: ModelProto) -> str:
    """Generate HTML representation of ONNX model."""
    lines: List[str] = []

    lines.append("<!DOCTYPE html>")
    lines.append("<html>")
    lines.append("<head>")
    lines.append('<meta charset="UTF-8">')
    lines.append("<title>ONNX Model</title>")
    lines.append("<style>")
    lines.append("ul, ol { list-style-type: disc; }")
    lines.append("</style>")
    lines.append("</head>")
    lines.append("<body>")

    graph = model.graph

    # Build value info map
    value_info_map: Dict[str, str] = {}
    for vi in graph.input:
        value_info_map[vi.name] = format_value_info(vi)
    for vi in graph.output:
        value_info_map[vi.name] = format_value_info(vi)
    for vi in graph.value_info:
        value_info_map[vi.name] = format_value_info(vi)
    for init in graph.initializer:
        dtype_name = get_tensor_type_name(init.data_type)
        shape = str(list(init.dims))
        value_info_map[init.name] = f"{init.name}: {dtype_name}{shape}"

    # Graph inputs
    lines.append("<h2>Graph Inputs</h2>")
    if graph.input:
        lines.append("<ul>")
        for inp in graph.input:
            value_link = f'<a href="#value-{escape(inp.name)}">{escape(inp.name)}</a>'
            type_shape = format_value_info(inp).split(": ", 1)
            if len(type_shape) > 1:
                lines.append(f"<li>{value_link}: {escape(type_shape[1])}</li>")
            else:
                lines.append(f"<li>{value_link}</li>")
        lines.append("</ul>")
    else:
        lines.append("<p>No inputs</p>")

    # Graph outputs
    lines.append("<h2>Graph Outputs</h2>")
    if graph.output:
        lines.append("<ul>")
        for out in graph.output:
            value_link = f'<a href="#value-{escape(out.name)}">{escape(out.name)}</a>'
            type_shape = format_value_info(out).split(": ", 1)
            if len(type_shape) > 1:
                lines.append(f"<li>{value_link}: {escape(type_shape[1])}</li>")
            else:
                lines.append(f"<li>{value_link}</li>")
        lines.append("</ul>")
    else:
        lines.append("<p>No outputs</p>")

    # Nodes
    lines.append(f"<h2>Nodes ({len(graph.node)})</h2>")
    if graph.node:
        lines.append("<ol>")

        for i, node in enumerate(graph.node):
            lines.append(f'<li id="node-{i}">')
            lines.append(f"<strong>{escape(node.op_type)}</strong>")

            if node.name:
                lines.append(f" ({escape(node.name)})")

            lines.append("<ul>")

            if node.input:
                lines.append("<li>Inputs:<ul>")
                for inp in node.input:
                    if inp:
                        value_link = f'<a href="#value-{escape(inp)}">{escape(inp)}</a>'
                        if inp in value_info_map:
                            # Extract type info from value_info_map
                            type_info = value_info_map[inp].split(": ", 1)
                            if len(type_info) > 1:
                                lines.append(
                                    f"<li>{value_link}: {escape(type_info[1])}</li>"
                                )
                            else:
                                lines.append(f"<li>{value_link}</li>")
                        else:
                            lines.append(f"<li>{value_link}</li>")
                lines.append("</ul></li>")

            if node.output:
                lines.append("<li>Outputs:<ul>")
                for out in node.output:
                    if out:
                        value_link = f'<a href="#value-{escape(out)}">{escape(out)}</a>'
                        if out in value_info_map:
                            # Extract type info from value_info_map
                            type_info = value_info_map[out].split(": ", 1)
                            if len(type_info) > 1:
                                lines.append(
                                    f"<li>{value_link}: {escape(type_info[1])}</li>"
                                )
                            else:
                                lines.append(f"<li>{value_link}</li>")
                        else:
                            lines.append(f"<li>{value_link}</li>")
                lines.append("</ul></li>")

            if node.attribute:
                lines.append("<li>Attributes:<ul>")
                for attr in node.attribute:
                    attr_value = format_attribute(attr)
                    lines.append(f"<li>{escape(attr.name)} = {escape(attr_value)}</li>")
                lines.append("</ul></li>")

            lines.append("</ul>")
            lines.append("</li>")

        lines.append("</ol>")
    else:
        lines.append("<p>No nodes</p>")

    # Values (all node inputs and outputs)
    all_values = set()
    value_producers: Dict[str, List[int]] = {}
    value_consumers: Dict[str, List[int]] = {}
    initializer_names = {init.name for init in graph.initializer}
    graph_input_names = {inp.name for inp in graph.input}
    graph_output_names = {out.name for out in graph.output}

    for i, node in enumerate(graph.node):
        for inp in node.input:
            if inp:
                all_values.add(inp)
                if inp not in value_consumers:
                    value_consumers[inp] = []
                value_consumers[inp].append(i)

        for out in node.output:
            if out:
                all_values.add(out)
                if out not in value_producers:
                    value_producers[out] = []
                value_producers[out].append(i)

    # Add initializers and graph inputs/outputs to all_values
    all_values.update(initializer_names)
    all_values.update(graph_input_names)
    all_values.update(graph_output_names)

    if all_values:
        lines.append(f"<h2>Values ({len(all_values)})</h2>")
        lines.append("<ul>")

        for value_name in sorted(all_values):
            lines.append(f'<li id="value-{escape(value_name)}">')

            # Value name and type info
            if value_name in value_info_map:
                lines.append(f"{escape(value_info_map[value_name])}")
            else:
                lines.append(f"{escape(value_name)}")

            # Producer and Consumer info
            lines.append("<ul>")

            # Producers
            if value_name in value_producers:
                producer_links = []
                for node_idx in value_producers[value_name]:
                    node = graph.node[node_idx]
                    node_label = (
                        node.name if node.name else f"{node.op_type}#{node_idx}"
                    )
                    producer_links.append(
                        f'<a href="#node-{node_idx}">{escape(node_label)}</a>'
                    )
                lines.append(f"<li>Producer: {', '.join(producer_links)}</li>")
            elif value_name in initializer_names:
                lines.append("<li>Producer: <strong>(initializer)</strong></li>")
            elif value_name in graph_input_names:
                lines.append("<li>Producer: <strong>(graph input)</strong></li>")
            else:
                lines.append("<li>Producer: (unknown)</li>")

            # Consumers
            consumer_parts = []
            if value_name in value_consumers:
                for node_idx in value_consumers[value_name]:
                    node = graph.node[node_idx]
                    node_label = (
                        node.name if node.name else f"{node.op_type}#{node_idx}"
                    )
                    consumer_parts.append(
                        f'<a href="#node-{node_idx}">{escape(node_label)}</a>'
                    )
            if value_name in graph_output_names:
                consumer_parts.append("<strong>(graph output)</strong>")

            if consumer_parts:
                lines.append(f"<li>Consumer: {', '.join(consumer_parts)}</li>")
            else:
                lines.append("<li>Consumer: (unused)</li>")

            lines.append("</ul>")
            lines.append("</li>")

        lines.append("</ul>")

    # Model metadata (at the end)
    lines.append("<h2>ONNX Model</h2>")
    lines.append("<ul>")

    if model.producer_name:
        lines.append(f"<li>Producer: {escape(model.producer_name)}</li>")

    if model.producer_version:
        lines.append(f"<li>Producer Version: {escape(model.producer_version)}</li>")

    if model.model_version:
        lines.append(f"<li>Model Version: {model.model_version}</li>")

    lines.append(f"<li>IR Version: {model.ir_version}</li>")

    if model.opset_import:
        opsets = ", ".join(
            f"{op.domain if op.domain else 'ai.onnx'}:{op.version}"
            for op in model.opset_import
        )
        lines.append(f"<li>Opset: {escape(opsets)}</li>")

    lines.append("</ul>")

    lines.append("</body>")
    lines.append("</html>")

    return "\n".join(lines)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert ONNX model to plain HTML for viewing in w3m"
    )
    parser.add_argument("input", help="Input ONNX model file")
    parser.add_argument(
        "-o", "--output", help="Output HTML file (default: stdout)", default=None
    )

    args = parser.parse_args()

    try:
        model = onnx.load(args.input)
    except Exception as e:
        print(f"Error loading ONNX model: {e}", file=sys.stderr)
        sys.exit(1)

    # Perform shape inference to get as much value info as possible
    try:
        print("Running shape inference...", file=sys.stderr)
        model = shape_inference.infer_shapes(model)
        print("Shape inference completed", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Shape inference failed: {e}", file=sys.stderr)
        print("Continuing without shape inference", file=sys.stderr)

    html_output = generate_html(model)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(html_output)
            print(f"HTML written to {args.output}", file=sys.stderr)
        except Exception as e:
            print(f"Error writing output file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(html_output)


if __name__ == "__main__":
    main()
