from lightgbm import LGBMRegressor
import numpy as np


class LGBMExpert:
    def __init__(
        self,
        features=None,
        alpha=0.65,
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=10,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
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
                "Wind_Dir_Meteo_cos",
            ]

        self.features = features
        self.alpha = alpha

        self.model = LGBMRegressor(
            objective="quantile",
            alpha=self.alpha,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1
        )

        self.is_fitted = False

    def fit(self, X, y):
        X_sel = X[self.features]
        y_log = np.log1p(y)
        self.model.fit(X_sel, y_log)
        self.is_fitted = True

    def predict(self, X):
        X_sel = X[self.features]
        y_log_pred = self.model.predict(X_sel)
        return np.expm1(y_log_pred)
