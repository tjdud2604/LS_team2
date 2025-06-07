# dashboard_admin.py

from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# ====== 전역 변수 ======
def get_default_thresholds_by_mold():
    return {
        "8722": round(float(np.float64(0.0531909483854564)) * 100, 2),
        "8412": round(float(np.float64(0.03438379900368205)) * 100, 2),
        "8573": round(float(np.float64(0.04376823676531888)) * 100, 2),
        "8917": round(float(np.float64(0.04484578807311434)) * 100, 2),
        "8600": round(float(np.float64(0.05067567567567568)) * 100, 2)
    }

DEFAULT_THRESHOLDS_BY_MOLD = get_default_thresholds_by_mold()
SENSORS = ['upper_mold_temp1','upper_mold_temp2','lower_mold_temp1','lower_mold_temp2',
           'cast_pressure','sleeve_temperature','low_section_speed','Coolant_temperature']
TRAIN_PATH = "./data/train.csv"

def compute_valid_ranges(data, threshold_dict):
    valid_ranges_by_mold = {}
    for mold_code in data["mold_code"].unique():
        df_mold = data[data["mold_code"] == mold_code]
        valid_ranges_by_mold[str(mold_code)] = {}
        for sensor in SENSORS:
            df_sensor = df_mold[[sensor, "passorfail"]].dropna()
            if df_sensor.empty or df_sensor[sensor].nunique() < 2:
                valid_ranges_by_mold[str(mold_code)][sensor] = []
                continue

            Q1 = df_sensor[sensor].quantile(0.25)
            Q3 = df_sensor[sensor].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_sensor = df_sensor[(df_sensor[sensor] >= lower) & (df_sensor[sensor] <= upper)]

            grouped = (
                df_sensor.groupby(sensor)
                .agg(total=('passorfail', 'count'), fail=('passorfail', lambda x: (x == 1).sum()))
                .reset_index()
                .sort_values(by=sensor)
            )
            grouped["fail_rate"] = grouped["fail"] / grouped["total"]

            threshold = threshold_dict.get(str(mold_code), 5.0) / 100
            ranges = []
            current_range = []

            for _, row in grouped.iterrows():
                if row["fail_rate"] < threshold:
                    current_range.append(row[sensor])
                else:
                    if current_range:
                        ranges.append((min(current_range), max(current_range)))
                        current_range = []
            if current_range:
                ranges.append((min(current_range), max(current_range)))

            valid_ranges_by_mold[str(mold_code)][sensor] = ranges
    return valid_ranges_by_mold

def ad_layout():
    data = pd.read_csv(TRAIN_PATH, low_memory=False)
    unique_molds = sorted(data['mold_code'].unique())
    default_mold = "8722"

    initial_valid_ranges = compute_valid_ranges(data, DEFAULT_THRESHOLDS_BY_MOLD)

    sensor_cards = [html.Div([
        html.Label(sensor),
        dcc.Graph(id=f"sensor-graph-{sensor}", style={"height": "300px"})
    ], style={"border": "1px solid lightgray", "padding": "10px", "borderRadius": "10px"}) for sensor in SENSORS]

    return html.Div([
        html.H3("확인용 - mold_code확인용"),
        dcc.Store(id="threshold-per-mold", data=DEFAULT_THRESHOLDS_BY_MOLD),
        dcc.Store(id="valid-ranges-store", data=initial_valid_ranges),

        html.Div([
            html.Label("mold_code", style={"fontWeight": "bold"}),
            dcc.Dropdown(id="mold-codes-dropdown",
                         options=[{"label": str(m), "value": str(m)} for m in unique_molds],
                         value=default_mold,
                         style={"width": "300px"})
        ], style={"marginBottom": "20px"}),

        html.Div([
            html.Label("확인용 (%)", style={"fontWeight": "bold"}),
            dcc.Slider(id="threshold-slider", min=0, max=10, step=0.01,
                       tooltip={"placement": "bottom", "always_visible": True})
        ], style={"marginBottom": "30px"}),

        html.Div(sensor_cards, style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "20px"}),
        html.Div(id="result-prob-card", style={"display": "none"}),
        html.Div(id="fault-record", style={"display": "none"}),
        html.Div(id="sensor-card-container", style={"display": "none"}),
        dcc.Graph(id="prob-graph", style={"display": "none"}),
        dcc.Dropdown(id="time-range-selector",style={"display": "none"})
    ])

def register_callbacks(app):
    @app.callback(
        Output("threshold-slider", "value"),
        Input("mold-codes-dropdown", "value"),
        State("threshold-per-mold", "data")
    )
    def update_slider_by_mold(mold_code, store):
        return store.get(mold_code, DEFAULT_THRESHOLDS_BY_MOLD.get(mold_code, 5.0))

    @app.callback(
        [Output(f"sensor-graph-{sensor}", "figure") for sensor in SENSORS],
        Input("mold-codes-dropdown", "value"),
        Input("threshold-slider", "value")
    )
    def update_graphs(selected_mold, slider_val):
        data = pd.read_csv(TRAIN_PATH, low_memory=False)
        mold_data = data[data['mold_code'] == int(selected_mold)]
        figures = []
        threshold_map = slider_val / 100

        for sensor in SENSORS:
            df = mold_data[[sensor, 'passorfail']].dropna()
            if df.empty or df[sensor].nunique() < 2:
                figures.append(go.Figure())
                continue

            Q1 = df[sensor].quantile(0.25)
            Q3 = df[sensor].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[sensor] >= lower) & (df[sensor] <= upper)]

            grouped = (
                df.groupby(sensor)
                .agg(total=('passorfail', 'count'),
                     fail=('passorfail', lambda x: (x == 1).sum()))
                .reset_index()
                .sort_values(by=sensor)
            )
            grouped['fail_rate'] = grouped['fail'] / grouped['total']

            fig = go.Figure()
            fig.add_trace(go.Bar(x=grouped[sensor], y=grouped['total'], name='총 생산 수', yaxis='y1'))
            fig.add_trace(go.Scatter(x=grouped[sensor], y=grouped['fail_rate'], name='불량률',
                                     mode='lines+markers', yaxis='y2', marker_color='red'))
            fig.add_hline(y=threshold_map, yref='y2', line=dict(color='green', dash='dash'),
                          annotation_text=f"기준선: {threshold_map:.2%}", annotation_position="bottom right")
            fig.update_layout(title=f"{sensor} 분석 (mold_code={selected_mold})",
                              xaxis=dict(title=sensor),
                              yaxis=dict(title='총 생산 수'),
                              yaxis2=dict(title='불량률', overlaying='y', side='right', range=[0, 0.1]))
            figures.append(fig)
        return figures

    @app.callback(
        Output("threshold-per-mold", "data"),
        Output("valid-ranges-store", "data"),
        Input("threshold-slider", "value"),
        State("mold-codes-dropdown", "value"),
        State("threshold-per-mold", "data"),
        prevent_initial_call=True
    )
    def update_threshold_store(threshold_val, mold_code, store):
        store = store or {}
        store[mold_code] = threshold_val
        data = pd.read_csv(TRAIN_PATH, low_memory=False)
        valid_ranges = compute_valid_ranges(data, store)
        return store, valid_ranges

__all__ = ["ad_layout", "register_callbacks"]
