"""Update the embedded model in logic.py with a newly trained checkpoint.

Usage:
    python update_model.py model.json
    python update_model.py model.json --no-backup
"""

import argparse
import json
import sys
from pathlib import Path


def find_dict_bounds(content: str, start_marker: str) -> tuple:
    """Find the start and end indices of a Python dict in source code."""
    start = content.find(start_marker)
    if start == -1:
        return -1, -1
    
    # Find the opening brace
    brace_start = content.find("{", start)
    if brace_start == -1:
        return -1, -1
    
    # Count braces to find the matching closing brace
    depth = 0
    i = brace_start
    while i < len(content):
        if content[i] == "{":
            depth += 1
        elif content[i] == "}":
            depth -= 1
            if depth == 0:
                return brace_start, i + 1
        i += 1
    
    return -1, -1


def update_logic_py(model_path: str, logic_path: str = "logic.py", backup: bool = True):
    """Replace the embedded model dict in logic.py with the new checkpoint."""
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)

    with open(model_path, "r") as f:
        new_model = json.load(f)

    logic_path = Path(logic_path)
    if not logic_path.exists():
        print(f"❌ logic.py not found: {logic_path}")
        sys.exit(1)

    content = logic_path.read_text()

    # Validate required keys
    required = ["feature_names", "mean", "std", "coef", "intercept"]
    for key in required:
        if key not in new_model:
            print(f"❌ Model missing required key: {key}")
            sys.exit(1)

    # Build the new model dict string
    model_str = json.dumps(new_model, indent=4)

    # Find the _EMBEDDED_MODEL dict
    start, end = find_dict_bounds(content, "_EMBEDDED_MODEL")
    if start == -1:
        print("❌ Could not find _EMBEDDED_MODEL in logic.py")
        sys.exit(1)

    # Replace the dict, keeping the type annotation
    # Find the colon before the dict
    colon_pos = content.rfind(":", 0, start)
    if colon_pos == -1:
        print("❌ Could not find type annotation for _EMBEDDED_MODEL")
        sys.exit(1)

    new_content = content[:colon_pos + 1] + " " + model_str + content[end:]

    if backup:
        backup_path = logic_path.with_suffix(".py.bak")
        logic_path.rename(backup_path)
        print(f"📦 Backed up original to {backup_path}")

    logic_path.write_text(new_content)
    print(f"✅ Updated {logic_path} with model from {model_path}")
    print(f"   Accuracy: {new_model.get('top1_accuracy', '?')}")


def main():
    parser = argparse.ArgumentParser(description="Update embedded model in logic.py")
    parser.add_argument("model", help="Path to model.json checkpoint")
    parser.add_argument("--logic", default="logic.py", help="Path to logic.py")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup")
    args = parser.parse_args()

    update_logic_py(args.model, args.logic, backup=not args.no_backup)


if __name__ == "__main__":
    main()
