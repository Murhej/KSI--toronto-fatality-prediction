# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 12:12:47 2025
@author: Murhej Hantoush / 301-325-315
"""

from pathlib import Path
from time import time
import os
import pickle
import joblib

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    fbeta_score,
    recall_score,
    make_scorer,
    precision_score,
    precision_recall_curve,
    auc,
    log_loss,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import StackingClassifier

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier

from xgboost import XGBClassifier
import lightgbm as lgb


# -------------------------
# -------------------------
# CONFIG
# -------------------------
HERE = Path(__file__).resolve()

# Project root = KSI--TORONTO-FATALITY-PREDICTION
PROJECT_ROOT = HERE.parents[1]   # because file is now Model/train_model.py

DATA_PATH = (
    PROJECT_ROOT
    / "Model"
    / "DataVisal"
    / "Traffic_Collisions_Open_Data_3719442797094142699.csv"
)

SAVE_DIR = PROJECT_ROOT / "Model"
JOBLIB_NAME = "Best_traffic_model.joblib"
PICKLE_NAME = "Best_traffic_model.pkl"

print("Project root:", PROJECT_ROOT)
print("CSV path:", DATA_PATH)
print("CSV exists:", DATA_PATH.exists())


# -------------------------
# CUSTOM SCORER
# -------------------------
def custom_scorer(y_true, y_pred):
    f2 = fbeta_score(y_true, y_pred, beta=2)
    rec = recall_score(y_true, y_pred)
    return 0.6 * f2 + 0.4 * rec

custom_scorer_obj = make_scorer(custom_scorer)


# -------------------------
# HELPERS
# -------------------------
class CyclicalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols_info=None):
        self.cols_info = cols_info or {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        for col, period in self.cols_info.items():
            vals = pd.to_numeric(X_[col], errors="coerce")
            X_[f"{col}_sin"] = np.sin(2 * np.pi * vals / period)
            X_[f"{col}_cos"] = np.cos(2 * np.pi * vals / period)
        X_.drop(columns=list(self.cols_info.keys()), inplace=True)
        return X_


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found at:\n{path}\n\n"
            f"Fix: put the CSV there OR update DATA_PATH in app.py."
        )
    return pd.read_csv(path)


def make_binary_target(df: pd.DataFrame, target_col="FATALITIES") -> pd.Series:
    return df[target_col].apply(lambda v: 0 if v == 0 else 1)


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    dow_mapping = {"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7}
    month_mapping = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
    }

    if "OCC_DOW" in df.columns and df["OCC_DOW"].dtype == "object":
        df["OCC_DOW"] = df["OCC_DOW"].map(dow_mapping)
    if "OCC_MONTH" in df.columns and df["OCC_MONTH"].dtype == "object":
        df["OCC_MONTH"] = df["OCC_MONTH"].map(month_mapping)
    if "OCC_HOUR" in df.columns:
        df["OCC_HOUR"] = pd.to_numeric(df["OCC_HOUR"], errors="coerce")

    cyc = CyclicalEncoder(cols_info={"OCC_DOW": 7, "OCC_MONTH": 12, "OCC_HOUR": 24})
    cyc_feats = cyc.transform(df[["OCC_DOW", "OCC_MONTH", "OCC_HOUR"]])
    return pd.concat([df, cyc_feats], axis=1)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # replace missing codes
    df.replace(["N/R", "NSA"], np.nan, inplace=True)

    conv_cols = ["AUTOMOBILE", "MOTORCYCLE", "PASSENGER", "BICYCLE", "PEDESTRIAN", "FTR_COLLISIONS", "PD_COLLISIONS"]
    for col in conv_cols:
        if col in df.columns:
            df[col] = df[col].map({"YES": 1, "NO": 0})

    vehicle_cols = ["AUTOMOBILE", "MOTORCYCLE", "PASSENGER", "BICYCLE", "PEDESTRIAN"]
    for c in vehicle_cols:
        if c not in df.columns:
            df[c] = 0

    df["vehicle_count"] = df[vehicle_cols].sum(axis=1)
    df["vulnerable_user"] = df[["PEDESTRIAN", "BICYCLE"]].max(axis=1)

    df["is_night"] = df["OCC_HOUR"].apply(lambda x: 1 if pd.notna(x) and (x >= 20 or x <= 5) else 0)
    df["rush_hour"] = df["OCC_HOUR"].apply(lambda x: 1 if pd.notna(x) and ((7 <= x <= 9) or (16 <= x <= 18)) else 0)

    df["season"] = df["OCC_MONTH"].apply(
        lambda m: "Winter" if m in [12, 1, 2] else "Spring" if m in [3, 4, 5] else "Summer" if m in [6, 7, 8] else "Fall"
    )

    if "DIVISION" in df.columns:
        df["division_ftr_rate"] = df["DIVISION"].map(df.groupby("DIVISION")["FTR_COLLISIONS"].mean())
        df["division_fatality_rate"] = df["DIVISION"].map(df.groupby("DIVISION")["FATALITIES"].mean())
        df["division_pd"] = df["DIVISION"].map(df.groupby("DIVISION")["PD_COLLISIONS"].mean())

    if "HOOD_158" in df.columns:
        df["division_Hood"] = df["HOOD_158"].map(df.groupby("HOOD_158")["PD_COLLISIONS"].mean())

    df["is_weekend"] = df["OCC_DOW"].isin([6, 7]).astype(int)
    df["night_weekend"] = df["is_night"] * df["is_weekend"]
    df["rush_hour_weekday"] = df["rush_hour"] * (1 - df["is_weekend"])

    # numeric cleanup for imputer & clustering
    for col in ["HOOD_158", "x", "y"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # iterative impute numeric cols
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 0:
        imp = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
        df[num_cols] = imp.fit_transform(df[num_cols])

    # location cluster if columns exist
    if all(c in df.columns for c in ["HOOD_158", "x", "y"]):
        df["location_cluster"] = KMeans(n_clusters=10, random_state=42).fit_predict(df[["HOOD_158", "y", "x"]])
        df["accident_density"] = df.groupby("location_cluster")["FATALITIES"].transform("mean")

    # rolling stats (requires OCC_DATE)
    if "OCC_DATE" in df.columns and "DIVISION" in df.columns:
        df = df.sort_values("OCC_DATE")
        df["rolling_accident_mean"] = df.groupby("DIVISION")["FATALITIES"].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )

    return df


def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return pre


def build_pipeline(model, preprocessor):
    return ImbPipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("remove_constant", VarianceThreshold(threshold=0.0)),
            ("feature_selection", SelectKBest(score_func=f_classif, k=20)),
            ("oversample", SMOTE(random_state=42, sampling_strategy=0.1)),
            ("undersample", RandomUnderSampler(random_state=42, sampling_strategy=0.7)),
            ("model", model),
        ]
    )


def main():
    # 1) load
    df = load_data(DATA_PATH)

    # 2) cyclical + features
    df = add_cyclical_features(df)
    df = feature_engineering(df)

    # 3) target
    y = make_binary_target(df, "FATALITIES")

    # 4) features (drop columns)
    drop_cols = [
        "FATALITIES", "OBJECTID", "EVENT_UNIQUE_ID", "FTR_COLLISIONS",
        "INJURY_COLLISIONS", "PD_COLLISIONS", "NEIGHBOURHOOD_158",
        "OCC_DATE", "OCC_MONTH", "OCC_DOW", "OCC_HOUR"
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # 5) split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 6) preprocessor
    preprocessor = build_preprocessor(X_train)

    # 7) models
    models = {
        "Balanced Random Forest": (
            BalancedRandomForestClassifier(random_state=42, n_estimators=500, class_weight="balanced_subsample"),
            {
                "model__n_estimators": [300, 500, 800],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        "XGBoost": (
            XGBClassifier(random_state=42, eval_metric="logloss", scale_pos_weight=10),
            {
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.1],
                "model__n_estimators": [100, 200, 500],
                "model__min_child_weight": [1, 3, 5],
            },
        ),
        "LightGBM": (
            lgb.LGBMClassifier(random_state=42, is_unbalance=True),
            {
                "model__num_leaves": [31, 63, 127],
                "model__learning_rate": [0.01, 0.1],
                "model__n_estimators": [100, 200, 500],
            },
        ),
    }

    results = {}

    # 8) train + tune
    for name, (model, params) in models.items():
        print(f"\n=== RandomizedSearchCV: {name} ===")
        pipe = build_pipeline(model, preprocessor)

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=params,
            n_iter=5,
            cv=3,
            scoring=custom_scorer_obj,
            n_jobs=-1,
            random_state=42,
            verbose=2,
            error_score="raise",
        )

        # smaller subset for speed
        X_train_sample, _, y_train_sample, _ = train_test_split(
            X_train, y_train, train_size=0.10, random_state=42, stratify=y_train
        )

        t0 = time()
        search.fit(X_train_sample, y_train_sample)
        print(f"Time: {time() - t0:.2f}s")

        best = search.best_estimator_
        y_proba = best.predict_proba(X_test)[:, 1]

        # choose threshold to maximize recall (like you were doing)
        thresholds = np.arange(0.1, 0.9, 0.01)
        recalls = [recall_score(y_test, (y_proba > t).astype(int)) for t in thresholds]
        best_thr = float(thresholds[int(np.argmax(recalls))])

        y_pred_best = (y_proba > best_thr).astype(int)
        f2 = fbeta_score(y_test, y_pred_best, beta=2)
        auroc = roc_auc_score(y_test, y_proba)

        print("AUROC:", auroc)
        print("Best recall-threshold:", best_thr)
        print("F2:", f2)
        print(classification_report(y_test, y_pred_best))

        results[name] = {"f2": f2, "threshold": best_thr, "model": best, "auroc": auroc}

    # 9) best model
    best_name = max(results, key=lambda k: results[k]["f2"])
    best_model = results[best_name]["model"]
    print(f"\n✅ Best model by F2: {best_name}")

    # 10) calibrate
    calibrated = CalibratedClassifierCV(best_model, cv=2, method="sigmoid")
    t0 = time()
    calibrated.fit(X_train, y_train)
    print(f"Calibration time: {time()-t0:.2f}s")

    y_proba_cal = calibrated.predict_proba(X_test)[:, 1]
    auroc_cal = roc_auc_score(y_test, y_proba_cal)
    print("AUROC after calibration:", auroc_cal)

    # 11) find best threshold for F2
    thresholds = np.arange(0.05, 0.95, 0.005)
    f2s = [fbeta_score(y_test, (y_proba_cal >= t).astype(int), beta=2) for t in thresholds]
    best_thr = float(thresholds[int(np.argmax(f2s))])

    y_pred_opt = (y_proba_cal >= best_thr).astype(int)

    print("\n=== Final Calibrated Performance ===")
    print("Best threshold (F2):", best_thr)
    print("Accuracy:", accuracy_score(y_test, y_pred_opt))
    print("Precision:", precision_score(y_test, y_pred_opt))
    print("Recall:", recall_score(y_test, y_pred_opt))
    print("F2:", fbeta_score(y_test, y_pred_opt, beta=2))
    print("Confusion:\n", confusion_matrix(y_test, y_pred_opt))
    print(classification_report(y_test, y_pred_opt))

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba_cal)
    pr_auc = auc(recall_vals, precision_vals)
    print("PR AUC:", pr_auc)
    print("Log Loss:", log_loss(y_test, y_proba_cal))

    model_config = {
        "model": calibrated,
        "threshold": {
            "optimal": best_thr,
            "high_risk": float(min(best_thr + 0.2, 0.99)),
            "low_risk": float(max(best_thr - 0.2, 0.01)),
        },
        "metadata": {
            "model_name": best_name,
            "metrics": {
                "auroc": float(auroc_cal),
                "f2_score": float(fbeta_score(y_test, y_pred_opt, beta=2)),
                "precision": float(precision_score(y_test, y_pred_opt)),
                "recall": float(recall_score(y_test, y_pred_opt)),
            },
        },
    }

    os.makedirs(SAVE_DIR, exist_ok=True)
    joblib_path = Path(SAVE_DIR) / JOBLIB_NAME
    pickle_path = Path(SAVE_DIR) / PICKLE_NAME

    joblib.dump(model_config, joblib_path)
    with open(pickle_path, "wb") as f:
        pickle.dump(model_config, f)

    print(f"\n✅ Saved:")
    print(" -", joblib_path)
    print(" -", pickle_path)


if __name__ == "__main__":
    main()
