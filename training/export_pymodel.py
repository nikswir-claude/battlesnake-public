"""Export the trained LightGBM booster to a dependency-free Python scorer.

Serving lightgbm+numpy on a 512MB free instance is slow to import (~8s) and
risks OOM. This flattens every tree into plain arrays saved as JSON; `logic.py`
then scores with a tiny pure-Python tree walk — no lightgbm, no numpy at serve.

    python export_pymodel.py --model ../model.txt --out ../model_py.json
"""

import argparse
import json
import os

import lightgbm as lgb


def flatten_tree(struct):
    """DFS a LightGBM tree_structure into parallel arrays (iterative-friendly)."""
    F, T, L, R, V = [], [], [], [], []

    def add(node):
        idx = len(F)
        F.append(-1); T.append(0.0); L.append(0); R.append(0); V.append(0.0)
        if "leaf_value" in node and "split_feature" not in node:
            V[idx] = float(node["leaf_value"])
        else:
            F[idx] = int(node["split_feature"])
            T[idx] = float(node["threshold"])
            L[idx] = add(node["left_child"])
            R[idx] = add(node["right_child"])
        return idx

    add(struct)
    return [F, T, L, R, V]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "model.txt"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_py.json"))
    args = ap.parse_args()

    booster = lgb.Booster(model_file=args.model)
    dump = booster.dump_model()
    trees = [flatten_tree(ti["tree_structure"]) for ti in dump["tree_info"]]
    out = {"feature_names": dump["feature_names"], "trees": trees}
    with open(args.out, "w") as f:
        json.dump(out, f)
    n_nodes = sum(len(t[0]) for t in trees)
    print(f"[export] trees={len(trees)} nodes={n_nodes} features={len(out['feature_names'])} "
          f"-> {args.out} ({os.path.getsize(args.out)//1024} KB)")


if __name__ == "__main__":
    main()
