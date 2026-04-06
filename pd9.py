# https://medium.com/@guyko81/stop-predicting-numbers-start-predicting-distributions-0d4975db52ae
# https://github.com/guyko81/DistributionRegressor



"""
Predicting Distributions - pd9
DistributionRegressor: Nonparametric Distributional Regression 
Lotto 7/39 probabilistic predictions
"""


import time
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
    LGBMRegressor = None


# -----------------------------
# Konfiguracija
# -----------------------------
SEED = 39
np.random.seed(SEED)

CSV_PATH = "data/loto7hh_4592_k27.csv"
COLS = ["Num1", "Num2", "Num3", "Num4", "Num5", "Num6", "Num7"]
FEATURE_COLS = [f"f{i+1}" for i in range(7)]

MIN_POS = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)
MAX_POS = np.array([33, 34, 35, 36, 37, 38, 39], dtype=int)

# Ukloni poznat warning koji zatrpava izlaz.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"sklearn\.utils\.validation",
    message=r"X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)


def load_draws(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if all(c in df.columns for c in COLS):
        arr = df[COLS].values.astype(float)
    else:
        arr = pd.read_csv(csv_path, header=None).iloc[:, :7].values.astype(float)
    return arr


def enforce_loto_7_39(nums_float: np.ndarray) -> np.ndarray:
    nums = np.rint(np.asarray(nums_float, dtype=float)).astype(int)
    nums = np.clip(nums, MIN_POS, MAX_POS)
    nums = np.sort(nums)

    for i in range(7):
        low = MIN_POS[i] if i == 0 else max(MIN_POS[i], nums[i - 1] + 1)
        nums[i] = max(nums[i], low)

    for i in range(6, -1, -1):
        high = MAX_POS[i] if i == 6 else min(MAX_POS[i], nums[i + 1] - 1)
        nums[i] = min(nums[i], high)

    for i in range(7):
        low = MIN_POS[i] if i == 0 else max(MIN_POS[i], nums[i - 1] + 1)
        nums[i] = max(nums[i], low)

    return nums


def make_builders():
    
    return [
        (
            "m5_soft_target",
            lambda: KNeighborsRegressor(
                n_neighbors=45,
                weights="distance",
                p=2,
            ),
        ),
        
    ]


def fit_predict_safe(model_name, build, X_df, y_vec, X_next_df):
    t0 = time.time()
    try:
        model = build()
        model.fit(X_df, y_vec)
        pred = float(np.asarray(model.predict(X_next_df)).ravel()[0])
        dt = time.time() - t0
        print(f"  - {model_name}: {pred:.6f}  (t={dt:.1f}s)")
        return pred
    except Exception as e:
        dt = time.time() - t0
        print(f"  - {model_name}: SKIP ({e})  (t={dt:.1f}s)")
        return None


def main():
    draws = load_draws(CSV_PATH)
    X = pd.DataFrame(draws[:-1], columns=FEATURE_COLS)
    Y = draws[1:]
    X_next = pd.DataFrame(draws[-1:].astype(float), columns=FEATURE_COLS)

    builders = make_builders()
    model_names = [name for name, _ in builders]
    per_model_raw = {name: [] for name in model_names}

    print("=" * 72)
    print("Prediction (Loto 7/39)")
    print("=" * 72)
    print(f"CSV: {CSV_PATH}")
    print(f"Uzoraka za trening: {len(X)}")
    print(f"Modela: {len(builders)}")
    print(f"LGBM dostupan: {'DA' if HAS_LGBM else 'NE (fallback aktivan)'}")
    print()

    # Trening i predikcija po poziciji
    for pos in range(7):
        y_pos = Y[:, pos]
        print(f"[pozicija {pos + 1}] trening + predikcija...")
        for name, build in builders:
            p = fit_predict_safe(name, build, X, y_pos, X_next)
            if p is not None:
                per_model_raw[name].append(p)
        print()

    # Posebna predikcija za svaki model
    print("=" * 72)
    print("PREDIKCIJA PO MODELU")
    print("=" * 72)
    per_model_final = {}
    for name in model_names:
        raw = per_model_raw.get(name, [])
        if len(raw) != 7:
            print(f"{name}: SKIP (nema svih 7 pozicija)")
            continue
        comb = enforce_loto_7_39(np.array(raw, dtype=float))
        per_model_final[name] = comb
        print(f"{name}: {comb}")
    print("=" * 72)
    


if __name__ == "__main__":
    main()

"""
========================================================================
Prediction (Loto 7/39)
========================================================================
CSV: /data/loto7hh_4592_k27.csv
Uzoraka za trening: 4591
Modela: 1
LGBM dostupan: DA

[pozicija 1] trening + predikcija...
  - m5_soft_target: 4.501884  (t=0.0s)

[pozicija 2] trening + predikcija...
  - m5_soft_target: 9.896982  (t=0.0s)

[pozicija 3] trening + predikcija...
  - m5_soft_target: x  (t=0.0s)

[pozicija 4] trening + predikcija...
  - m5_soft_target: y  (t=0.0s)

[pozicija 5] trening + predikcija...
  - m5_soft_target: z  (t=0.0s)

[pozicija 6] trening + predikcija...
  - m5_soft_target: 30.915698  (t=0.0s)

[pozicija 7] trening + predikcija...
  - m5_soft_target: 35.519957  (t=0.0s)

========================================================================
PREDIKCIJA PO MODELU
========================================================================
m5_soft_target: [ 5 10 x y z 31 36]
========================================================================
"""
