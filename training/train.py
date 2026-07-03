"""Train a LightGBM move-ranker on self-play data and benchmark it.

Label is "is this the expert's chosen move" (one positive per state); at serve
time we score every legal move and take the argmax. Evaluation uses a
group-wise split (no state leaks between train/val) and reports the metric that
actually matters — top-1 move accuracy — against the old linear baseline.

    python train.py --data data.npz --out ../model.txt
"""

import argparse
import json
import os

import lightgbm as lgb
import numpy as np

# Old embedded linear model (baseline). Its features are a subset of the new set.
_LINEAR = {
    "feature_names": ["space_capped", "open_space", "voronoi", "reaches_tail", "escape",
                      "h2h_danger", "near_bigger_head", "near_enemy_head", "wall_dist",
                      "food_score", "food_delta", "is_food", "dist_to_center"],
    "mean": [7.357954545454546, 100.9034090909091, 48.26988636363637, 0.9943181818181818,
             2.4431818181818183, 0.04261363636363636, 9.673295454545455, 4.676136363636363,
             1.625, 0.8920454545454546, 0.14772727272727273, 0.036931818181818184, 5.056818181818182],
    "std": [3.5995966185276513, 22.80542174802676, 31.41119158524981, 0.07516338951888041,
            0.6235520417417705, 0.20198444088469822, 7.9675173248507924, 2.2532045017839604,
            1.3552297691803878, 5.861056404757769, 0.9449599886584031, 0.18859442989548575, 2.34451950177747],
    "coef": [0.00010539398521136327, -1.6778512168946185, 80.89420182766183, 9.793855564450467,
             0.7884630868036275, -11.025170822665032, -0.7981723553489, 0.5410534990053248,
             1.5629078731518526, 7.582325762611304, 0.12463070008097832, 0.21036618806863483, 1.836259515524985],
    "intercept": 0.0,
}


def top1_accuracy(scores, y, groups):
    """Fraction of states where the argmax-scored move is the expert's move."""
    _, inv = np.unique(groups, return_inverse=True)
    ng = inv.max() + 1
    best_score = np.full(ng, -np.inf)
    best_idx = np.zeros(ng, dtype=np.int64)
    for i in range(len(inv)):
        g = inv[i]
        if scores[i] > best_score[g]:
            best_score[g] = scores[i]
            best_idx[g] = i
    return float(y[best_idx].mean())


def linear_scores(X, names):
    idx = {n: i for i, n in enumerate(names)}
    m = _LINEAR
    cols = [idx[n] for n in m["feature_names"]]
    Z = (X[:, cols] - np.array(m["mean"])) / np.array(m["std"])
    return Z @ np.array(m["coef"]) + m["intercept"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "data.npz"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "model.txt"))
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, y, groups = d["X"], d["y"].astype(np.int32), d["groups"]
    names = [str(s) for s in d["feature_names"]]
    print(f"[train] rows={len(y)} states={len(np.unique(groups))} features={len(names)} "
          f"pos_rate={y.mean():.3f}")

    # Group-wise split so no state appears in both train and val.
    uniq = np.unique(groups)
    rng = np.random.RandomState(42)
    rng.shuffle(uniq)
    n_val = int(len(uniq) * args.val_frac)
    val_groups = set(uniq[:n_val].tolist())
    is_val = np.array([g in val_groups for g in groups])
    Xtr, ytr, gtr = X[~is_val], y[~is_val], groups[~is_val]
    Xva, yva, gva = X[is_val], y[is_val], groups[is_val]
    print(f"[train] train_rows={len(ytr)} val_rows={len(yva)}")

    dtrain = lgb.Dataset(Xtr, label=ytr, feature_name=names)
    dval = lgb.Dataset(Xva, label=yva, reference=dtrain)
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "num_threads": os.cpu_count(),
    }
    booster = lgb.train(
        params, dtrain, num_boost_round=800, valid_sets=[dval],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(50)],
    )

    # Benchmarks on the val split.
    gbdt_va = booster.predict(Xva)
    lin_va = linear_scores(Xva, names)
    rng2 = np.random.RandomState(0)
    rand_va = rng2.random(len(yva))

    acc_gbdt = top1_accuracy(gbdt_va, yva, gva)
    acc_lin = top1_accuracy(lin_va, yva, gva)
    acc_rand = top1_accuracy(rand_va, yva, gva)
    print("\n=== top-1 move accuracy (val, matches expert) ===")
    print(f"  random baseline : {acc_rand:.4f}")
    print(f"  old linear model: {acc_lin:.4f}")
    print(f"  NEW LightGBM    : {acc_gbdt:.4f}   (+{acc_gbdt - acc_lin:.4f} vs linear)")

    print("\n=== feature importance (gain) ===")
    imp = sorted(zip(names, booster.feature_importance("gain")), key=lambda t: -t[1])
    tot = sum(v for _, v in imp) or 1.0
    for n, v in imp:
        print(f"  {n:20s} {100 * v / tot:5.1f}%")

    booster.save_model(args.out, num_iteration=booster.best_iteration)
    meta = {
        "feature_names": names,
        "best_iteration": int(booster.best_iteration),
        "top1_gbdt": acc_gbdt, "top1_linear": acc_lin, "top1_random": acc_rand,
    }
    with open(args.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[train] saved -> {args.out}  (best_iter={booster.best_iteration})")


if __name__ == "__main__":
    main()
