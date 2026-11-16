"""Read-only audit of DIOR XML boxes for bounds and positive dimensions."""

import argparse
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Directory containing DIOR Pascal VOC XML files",
    )
    parser.add_argument("--limit", type=int, default=20,
                        help="Maximum examples to print per issue (default: 20)")
    args = parser.parse_args()

    xml_files = sorted(args.annotations.glob("*.xml"))
    if not xml_files:
        raise SystemExit(
            f"No XML files found in {args.annotations}. Pass the actual "
            "Annotations/Horizontal Bounding Boxes directory."
        )
    issue_counts = Counter()
    examples = {"missing_size": [], "malformed": [], "out_of_bounds": [], "degenerate": []}
    boxes_checked = 0

    for xml_path in xml_files:
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        if size is None:
            issue_counts["missing_size"] += 1
            if len(examples["missing_size"]) < args.limit:
                examples["missing_size"].append(xml_path.name)
            continue
        width = int(size.findtext("width", default="0"))
        height = int(size.findtext("height", default="0"))

        for object_index, obj in enumerate(root.findall("object"), start=1):
            boxes_checked += 1
            bbox = obj.find("bndbox")
            try:
                coords = [float(bbox.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")]
            except (AttributeError, TypeError, ValueError):
                issue_counts["malformed"] += 1
                if len(examples["malformed"]) < args.limit:
                    examples["malformed"].append(f"{xml_path.name} object {object_index}")
                continue

            xmin, ymin, xmax, ymax = coords
            if xmax <= xmin or ymax <= ymin:
                issue_counts["degenerate"] += 1
                if len(examples["degenerate"]) < args.limit:
                    examples["degenerate"].append(f"{xml_path.name} object {object_index}: {coords}")
            if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                issue_counts["out_of_bounds"] += 1
                if len(examples["out_of_bounds"]) < args.limit:
                    examples["out_of_bounds"].append(
                        f"{xml_path.name} object {object_index}: {coords} vs {width}x{height}"
                    )

    print(f"XML files scanned: {len(xml_files)}")
    print(f"Boxes checked: {boxes_checked}")
    for issue in examples:
        print(f"{issue}: {issue_counts[issue]}")
        for example in examples[issue]:
            print(f"  {example}")
if __name__ == "__main__":
    main()
