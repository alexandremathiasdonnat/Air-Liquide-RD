import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd

class ElasticNetExpert():
    def __init__(
        self,
        features=None,
        alphas=np.logspace(-4, 0, 10),
        l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
        n_split=3,
        max_iter=5000,
        weight_power=1.2
    ):
        if features is None:
            features = [
                "speed_longitudinale_100m",
                "speed_latitudinale_100m",
                "2m_temperature",
                "sea_surface_temperature",
                "surface_pressure",
                "Wind_Norm",
                "Wind_Norm_Cubes",
                "wind_std_3h",
                "wind_cv_3h",
                "Wind_Norm_lag_1h",
                "Wind_Norm_lag_24h",
                "Hour_sin",
                "Hour_cos",
                "Weekday_sin",
                "Weekday_cos",
                "Month_sin",
                "Month_cos",
                "Wind_mean_3h",
                "Air_density",
                "Wind_Dir_Meteo_sin",
                "Wind_Dir_Meteo_cos"
            ]

        self.features = features
        self.weight_power = weight_power

        self.alphas = alphas if alphas is not None else np.logspace(-4, 2, 60)
        self.l1_ratios = l1_ratios if l1_ratios is not None else np.linspace(0.05, 0.9, 10)
        self.cv = TimeSeriesSplit(n_splits=n_split)

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("enet", ElasticNetCV(
                alphas=self.alphas,
                l1_ratio=self.l1_ratios,
                cv=self.cv,
                max_iter=max_iter,
                fit_intercept=True
            ))
        ])

        self.is_fitted = False
        self.calibration_ = None   # correction finale

    # -------------------------
    # Fit avec :
    # - log(target)
    # - sample_weight
    # -------------------------
    def fit(self, X, y):
        X_sel = X[self.features].copy()
        y = y.copy()

        # -------------------------
        # 1. Nettoyage target
        # -------------------------
        mask = y.notna() & (y >= 0)

        X_sel = X_sel.loc[mask]
        y = y.loc[mask]

        # sécurité ultime
        y = y.clip(lower=0)

        # -------------------------
        # 2. log transform
        # -------------------------
        y_log = np.log1p(y)

        # -------------------------
        # 3. sample weights
        # -------------------------
        weights = (y / (y.max() + 1e-6)) ** self.weight_power
        weights = np.clip(weights, 0.1, None)

        # alignement final
        weights = weights.loc[y.index]

        # -------------------------
        # 4. Fit pipeline
        # -------------------------
        self.pipeline.fit(
            X_sel,
            y_log,
            enet__sample_weight=weights
        )

        # -------------------------
        # 5. Calibration post-fit
        # -------------------------
        y_log_pred = self.pipeline.predict(X_sel)
        a, b = np.polyfit(y_log_pred, y_log, 1)
        self.calibration_ = (a, b)

        self.is_fitted = True

    # -------------------------
    # Predict inverse log + calibration
    # -------------------------
    def predict(self, X):
        X_sel = X[self.features]
        y_log_pred = self.pipeline.predict(X_sel)

        if self.calibration_ is not None:
            a, b = self.calibration_
            y_log_pred = a * y_log_pred + b

        return np.expm1(y_log_pred)

    # -------------------------
    # Diagnostics utiles
    # -------------------------
    @property
    def best_alpha_(self):
        return self.pipeline.named_steps["enet"].alpha_

    @property
    def best_l1_ratio_(self):
        return self.pipeline.named_steps["enet"].l1_ratio_

    def get_coefficients(self):
        enet = self.pipeline.named_steps["enet"]
        return enet.coef_, enet.intercept_