from sklearn.ensemble import RandomForestRegressor
import warnings
import numpy as np
warnings.filterwarnings("ignore")


class RandomForestExpert():
    def __init__(
        self,
        features=None,
        n_estimators=200,
        max_depth=12,
        min_samples_split=20,
        min_samples_leaf=20,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        weight_power=1.3
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

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=n_jobs
        )

        self.is_fitted = False
        self.calibration_ = None

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

        # Sécurité ultime
        y = y.clip(lower=0)

        # -------------------------
        # 2. log transform
        # -------------------------
        y_log = np.log1p(y)

        # -------------------------
        # 3. Sample weights
        # -------------------------
        weights = (y / (y.max() + 1e-6)) ** self.weight_power
        weights = np.clip(weights, 0.1, None)

        # Alignement final
        weights = weights.loc[y.index]

        # -------------------------
        # 4. Fit modèle
        # -------------------------
        self.model.fit(
            X_sel,
            y_log,
            sample_weight=weights
        )

        # -------------------------
        # 5. Calibration
        # -------------------------
        y_log_pred = self.model.predict(X_sel)
        a, b = np.polyfit(y_log_pred, y_log, 1)
        self.calibration_ = (a, b)

        self.is_fitted = True

    # -------------------------
    # Predict inverse log + calibration
    # -------------------------
    def predict(self, X):
        X_sel = X[self.features]
        y_log_pred = self.model.predict(X_sel)

        if self.calibration_ is not None:
            a, b = self.calibration_
            y_log_pred = a * y_log_pred + b

        return np.expm1(y_log_pred)