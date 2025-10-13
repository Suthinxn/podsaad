import numpy as np
import pandas as pd

ALPHA = 0.7
BETA_HT = 0.3
NOISE_STD_RATIO = 0.1

def generate_exog_future(exog_history, exog_cols, future_dates, LOOKBACK):
    exog_future_list = []
    exog_history = exog_history.copy()

    for i in range(len(future_dates)):
        next_values = {}
        current_date = future_dates[i]

        np.random.seed(int(current_date.strftime("%Y%m%d")))

        month = current_date.month

        col = 'humidity'
        last_values = exog_history[col].iloc[-LOOKBACK:].fillna(exog_history[col].mean())
        seasonal_values = exog_history[exog_history.index.month == month][col]
        seasonal_mean = seasonal_values.mean() if len(seasonal_values) > 0 else last_values.mean()
        noise = np.random.normal(0, last_values.std() * NOISE_STD_RATIO)
        humidity_next = np.clip(seasonal_mean + noise, 0, 100)
        next_values[col] = float(humidity_next)

        col = 'temperature'
        seasonal_values = exog_history[exog_history.index.month == month][col]
        seasonal_mean = seasonal_values.mean() if len(seasonal_values) > 0 else exog_history[col].iloc[-LOOKBACK:].mean()
        humidity_month_mean = exog_history['humidity'][exog_history.index.month == month].mean()
        temp_next = seasonal_mean - BETA_HT * (humidity_next - humidity_month_mean)
        next_values[col] = float(temp_next)

        for col in ['PM_1', 'PM_0_1', 'pressure']:
            last_values = exog_history[col].iloc[-LOOKBACK:].fillna(exog_history[col].mean())
            seasonal_values = exog_history[exog_history.index.month == month][col]
            seasonal_mean = seasonal_values.mean() if len(seasonal_values) > 0 else last_values.mean()
            prev_val = last_values.iloc[-1]
            noise = np.random.normal(0, last_values.std() * NOISE_STD_RATIO)
            next_val = ALPHA * prev_val + (1 - ALPHA) * seasonal_mean + noise
            if col.startswith("PM"):
                next_val = max(next_val, 0)
            next_values[col] = float(next_val)

        dow = current_date.weekday()
        next_values['day_of_week_sin'] = np.sin(2 * np.pi * dow / 7)
        next_values['day_of_week_cos'] = np.cos(2 * np.pi * dow / 7)
        next_values['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
        next_values['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)

        exog_history.loc[current_date] = next_values
        exog_future_list.append([next_values[c] for c in exog_cols])

    exog_future = pd.DataFrame(exog_future_list, columns=exog_cols, index=future_dates)
    exog_future = exog_future.astype(float)
    return exog_future
