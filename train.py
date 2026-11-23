"""CLI training:  python train.py"""
import os, json, pickle
import numpy as np, pandas as pd
from data_generator import generate
from features import to_matrix, TARGET
from model import GDModel, metrics, closed_form
from sklearn.model_selection import train_test_split

def split(df, test=0.2, seed=1):
    return train_test_split(df, test_size=test, random_state=seed)

def run(df, **hp):
    tr, te = split(df)
    Xtr, ytr = to_matrix(tr), tr[TARGET].to_numpy()
    Xte, yte = to_matrix(te), te[TARGET].to_numpy()
    m = GDModel(**hp).fit(Xtr, ytr, Xte, yte)
    out = dict(model=m, train=metrics(ytr, m.predict(Xtr)),
               test=metrics(yte, m.predict(Xte)), te=te,
               y_pred=m.predict(Xte))
    wn, bn = closed_form(Xtr, ytr)
    wg, bg = m.coefficients()
    out["max_coef_diff_vs_closed_form"] = float(np.max(np.abs(wg - wn)))
    return out

if __name__ == "__main__":
    if not os.path.exists("house_data.csv"):
        generate().to_csv("house_data.csv", index=False)
    df = pd.read_csv("house_data.csv")
    res = run(df, lr=0.01, epochs=100)
    print("Train:", res["train"]); print("Test :", res["test"])
    print("Max |GD - LinearRegression| coefficient gap:", res["max_coef_diff_vs_closed_form"])
    print("Epochs run:", len(res["model"].history["train_loss"]))
    pickle.dump(res["model"], open("model.pkl", "wb"))
