"""Linear Regression trained with Gradient Descent using scikit-learn.
SGDRegressor = linear model optimised by (stochastic) gradient descent.
StandardScaler = feature scaling.  sklearn.metrics = evaluation."""
import numpy as np
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

class GDModel:
    def __init__(self, lr=0.01, epochs=30, alpha=0.0, schedule="invscaling", seed=0):
        self.epochs = epochs
        self.xs, self.ys = StandardScaler(), StandardScaler()
        self.sgd = SGDRegressor(loss="squared_error", penalty="l2", alpha=alpha,
                                learning_rate=schedule, eta0=lr, shuffle=True,
                                random_state=seed)
        self.history = {"train_loss": [], "val_loss": []}

    def fit(self, X, y, X_val=None, y_val=None):
        Xs = self.xs.fit_transform(X)
        yn = self.ys.fit_transform(y.reshape(-1, 1)).ravel()
        if X_val is not None:
            Xv = self.xs.transform(X_val)
            yv = self.ys.transform(y_val.reshape(-1, 1)).ravel()
        self.history = {"train_loss": [], "val_loss": []}
        for _ in range(self.epochs):              # 1 epoch = 1 pass over data
            self.sgd.partial_fit(Xs, yn)
            self.history["train_loss"].append(mean_squared_error(yn, self.sgd.predict(Xs)) / 2)
            if X_val is not None:
                self.history["val_loss"].append(mean_squared_error(yv, self.sgd.predict(Xv)) / 2)
        return self

    @property
    def w(self):                                  # standardized weights
        return self.sgd.coef_

    def predict(self, X):
        p = self.sgd.predict(self.xs.transform(X))
        return self.ys.inverse_transform(p.reshape(-1, 1)).ravel()

    def coefficients(self):
        """Weights in original units ($ per unit) + intercept."""
        coef = self.sgd.coef_ * self.ys.scale_[0] / self.xs.scale_
        b = (self.ys.mean_[0] + self.sgd.intercept_[0] * self.ys.scale_[0]
             - float(np.sum(coef * self.xs.mean_)))
        return coef, b

def metrics(y, p):
    """Only MSE (Mean Squared Error) is reported."""
    return dict(MSE=float(mean_squared_error(y, p)))

def closed_form(X, y):
    """sklearn LinearRegression (exact solution) - used to verify GD."""
    lr = LinearRegression().fit(X, y)
    return lr.coef_, lr.intercept_
