# -*- coding: utf-8 -*-
import os

# Get the directory containing the current notebook
notebook_dir = os.path.dirname(os.path.abspath("__file__"))

# Change working directory to notebook folder
os.chdir(notebook_dir)

# Verify
print("Current working directory:", os.getcwd())

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder  # sklearn's TargetEncoder
import optuna
from sklearn.metrics import cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
# %%


# -----------------------------
# Load cleaned data
# -----------------------------
train = pd.read_csv('data/data_cleaned/train_clean.csv')
holdout = pd.read_csv('data/data_cleaned/holdout_cleaned.csv')

y_train = train['damage_grade']
X_train = train.drop(columns=['damage_grade'])

y_holdout = holdout['damage_grade']
X_holdout = holdout.drop(columns=['damage_grade'])

# Features to target‐encode
geo_target = ['geo__geo_level_2_id', 'geo__geo_level_3_id']


# %%


# -----------------------------
# Define Optuna objective
# -----------------------------
def objective(trial):
    # Hyperparameters for XGBoost
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 2, 20),
        "weights": trial.suggest_categorical("weigths", ["uniform", "distance"]),
        "p": trial.suggest_int("p", 1, 2)
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr = X_train.iloc[train_idx].copy()
        X_val = X_train.iloc[val_idx].copy()
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        # Use sklearn TargetEncoder
        te = TargetEncoder(cv=5, shuffle=True, random_state=42, target_type = "multiclass")
        # Note: fit_transform does *cross‑fitting* internally in sklearn version.
        X_tr_enc = te.fit_transform(X_tr[geo_target], y_tr)
        X_val_enc = te.transform(X_val[geo_target])

        # Replace original geo columns with encoded ones
        X_tr_enc = pd.DataFrame(X_tr_enc, index=X_tr.index, columns=te.get_feature_names_out(geo_target))
        X_val_enc = pd.DataFrame(X_val_enc, index=X_val.index, columns=te.get_feature_names_out(geo_target))

        X_tr_full = X_tr.copy()
        X_val_full = X_val.copy()
        X_tr_full.drop(columns=geo_target, inplace=True)
        X_val_full.drop(columns=geo_target, inplace=True)
        # concat encoded
        X_tr_full = pd.concat([X_tr_full, X_tr_enc], axis=1)
        X_val_full = pd.concat([X_val_full, X_val_enc], axis=1)

        # Train model
        model = KNeighborsClassifier(**params, n_jobs=1)
        model.fit(X_tr_full, y_tr)

        y_pred = model.predict(X_val_full)

        acc = cohen_kappa_score(y_pred, y_val, weights= "quadratic")
        cv_scores.append(acc)

    # Return mean accuracy so Optuna maximizes it
    return np.mean(cv_scores)

# -----------------------------
# Run Optuna
# -----------------------------
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=2, show_progress_bar=True, n_jobs=1)

print("Best params:", study.best_params)
print("Best CV accuracy:", study.best_value)

# %%

# -----------------------------
# Train final model on full train set with target encoding
# -----------------------------
te_final = TargetEncoder(cv=5, shuffle=True, random_state=42)
X_train_enc = te_final.fit_transform(X_train[geo_target], y_train)
X_holdout_enc = te_final.transform(X_holdout[geo_target])

X_train_enc = pd.DataFrame(X_train_enc, index=X_train.index, columns=te_final.get_feature_names_out(geo_target))
X_holdout_enc = pd.DataFrame(X_holdout_enc, index=X_holdout.index, columns=te_final.get_feature_names_out(geo_target))

X_train_full = X_train.copy()
X_holdout_full = X_holdout.copy()
X_train_full.drop(columns=geo_target, inplace=True)
X_holdout_full.drop(columns=geo_target, inplace=True)

X_train_full = pd.concat([X_train_full, X_train_enc], axis=1)
X_holdout_full = pd.concat([X_holdout_full, X_holdout_enc], axis=1)


final_model = KNeighborsClassifier(**study.best_params)
final_model.fit(X_train_full, y_train)

# -----------------------------
# Evaluate on holdout
# -----------------------------
holdout_acc = final_model.score(X_holdout_full, y_holdout)
print("Holdout accuracy:", holdout_acc)

# %%

import pickle

with open("artifacts/final_model_xgb.pkl", "wb") as f:
    pickle.dump(final_model, f)



