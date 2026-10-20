"""Train the pass difficulty model on simulated matches and report how well it does.

    python scripts/train_pass_model.py [n_matches]

Matches are simulated in parallel with varied storylines. The last three
matches are held out for testing. The model is judged on held-out completion
(AUC, Brier score, calibration) and on agreement with the simulator's own
ground-truth pass probabilities, which the model never sees.
"""

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from matchsense.analytics.context import MatchContext
from matchsense.analytics.passing import FEATURES, PassDifficultyModel, auc, fit_logistic, pass_features
from matchsense.sim import load_default_teams, simulate

STORIES = ("balanced", "comeback", "siege", "late_drama")


def extract(seed: int):
    home, away = load_default_teams()
    result = simulate(home, away, seed=seed, storyline=STORIES[seed % len(STORIES)])
    ctx = MatchContext(result.metadata)
    truth = {t["event"]: t["p_success"] for t in result.truth if t["type"] == "pass"}
    X, y, tp = [], [], []
    for ev in result.events:
        if ev["type"] != "pass" or ev["outcome"] == "offside":
            continue
        X.append(pass_features(ev, result.tracking["xy"][ev["frame"]], ctx))
        y.append(1.0 if ev["outcome"] == "complete" else 0.0)
        tp.append(truth.get(ev["id"], np.nan))
    return np.array(X), np.array(y), np.array(tp)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    seeds = list(range(5000, 5000 + n))
    with ProcessPoolExecutor() as pool:
        parts = list(pool.map(extract, seeds))
    train, test = parts[:-3], parts[-3:]
    Xtr = np.vstack([p[0] for p in train])
    ytr = np.concatenate([p[1] for p in train])
    Xte = np.vstack([p[0] for p in test])
    yte = np.concatenate([p[1] for p in test])
    tte = np.concatenate([p[2] for p in test])

    coef = fit_logistic(Xtr, ytr)
    model = PassDifficultyModel(coef, {})
    p = model.p_complete(Xte)
    bins = np.clip((p * 5).astype(int), 0, 4)
    calibration = [
        {"predicted": round(float(p[bins == b].mean()), 3), "actual": round(float(yte[bins == b].mean()), 3),
         "n": int((bins == b).sum())}
        for b in range(5) if (bins == b).any()
    ]
    ok = np.isfinite(tte)
    model.metrics = {
        "train_matches": len(train), "test_matches": len(test),
        "train_passes": int(len(ytr)), "test_passes": int(len(yte)),
        "test_auc": round(auc(yte, p), 3),
        "test_brier": round(float(np.mean((p - yte) ** 2)), 4),
        "baseline_brier": round(float(np.mean((yte.mean() - yte) ** 2)), 4),
        "test_completion": round(float(yte.mean()), 3),
        "mean_predicted": round(float(p.mean()), 3),
        "corr_with_simulator_truth": round(float(np.corrcoef(p[ok], tte[ok])[0, 1]), 3),
        "calibration": calibration,
    }
    model.save()
    print("coefficients:")
    for name, c in zip(FEATURES, coef):
        print(f"  {name:<10} {c:+.3f}")
    print("metrics:")
    for k, v in model.metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
