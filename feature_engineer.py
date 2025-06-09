from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

# === FeatureEngineer 정의 ===
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.left_threshold = None
        self.outlier_bounds = {}

    def fit(self, X, y=None):
        X = X.copy()
        if 'passorfail' in X.columns:
            X = X.drop(columns=['passorfail'])

        q1 = X['molten_temp'].quantile(0.25)
        q3 = X['molten_temp'].quantile(0.75)
        iqr = q3 - q1
        self.left_threshold = q1 - 3 * iqr

        for col in ['biscuit_thickness', 'sleeve_temperature', 'physical_strength', 'Coolant_temperature']:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.outlier_bounds[col] = (q1 - 3 * iqr, q3 + 3 * iqr)
        return self

    def transform(self, X):
        X = X.copy()

        X['molten_temp_is_left_outlier'] = (X['molten_temp'] < self.left_threshold).astype(int)
        X['molten_temp'] = X['molten_temp'].fillna(X['molten_temp'].median())

        X['low_section_speed_is_abnormal'] = ((X['low_section_speed'] <= 50) | (X['low_section_speed'] >= 1000)).astype(int)
        X['high_section_speed_is_abnormal'] = ((X['high_section_speed'] <= 50) | (X['high_section_speed'] >= 200)).astype(int)
        X['cast_pressure_is_low'] = (X['cast_pressure'] <= 300).astype(int)

        for col, (low, high) in self.outlier_bounds.items():
            X[f'{col}_is_outlier'] = ((X[col] < low) | (X[col] > high)).astype(int)

        if 'heating_furnace' in X.columns:
            X['heating_furnace'] = X['heating_furnace'].fillna('C')

        X['timestamp'] = pd.to_datetime(X['date'].astype(str) + ' ' + X['time'].astype(str), errors='coerce')
        X['hour'] = X['timestamp'].dt.hour.fillna(0)
        X['hour'] = X['hour'].infer_objects(copy=False)
        X['hour'] = X['hour'].astype(int)

        drop_cols = ["id", "line", "name", "mold_name", "count", "tryshot_signal", "emergency_stop", "working",
                     "time", "date", "upper_mold_temp3", "lower_mold_temp3", "registration_time", "timestamp"]
        X = X.drop(columns=drop_cols, errors='ignore')
        return X