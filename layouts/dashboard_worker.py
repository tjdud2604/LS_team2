# dashboard_worker.py

from dash import html, dcc, Output, Input, State, callback_context, no_update
import pandas as pd
import plotly.graph_objs as go
import joblib
import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import ALL
import os
import json
import sys
import numpy as np # numpy import 추가
from datetime import datetime, timedelta
from feature_engineer import FeatureEngineer
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)


DATA_PATH = "./data/answer.csv"

sensor_features = [
    "upper_mold_temp1", "upper_mold_temp2", "lower_mold_temp1", "lower_mold_temp2",
    "cast_pressure", "sleeve_temperature", "low_section_speed", "high_section_speed"
]
# sensor_thresholds = {
#     "upper_mold_temp1": [(99,104),(154, 999)],
#     "upper_mold_temp2": (154, 999),
#     "lower_mold_temp1": [(124,134),(159,309)],
#     "lower_mold_temp2": [(139,149),(169,999)],
#     "low_section_speed": [105, 110, 115],
#     "high_section_speed": [(101.5, 102.5),(109,113),(117,118)],
#     "cast_pressure": [(311,315),(318,334)],
#     "sleeve_temperature": [(160, 210), (460, 530)]
# }
SPECIALIZED_MODELS = {
    8412: "./data/model_8412.pkl",
    8573: "./data/model_8573.pkl",
    8600: "./data/model_8600.pkl",
    8722: "./data/model_8722.pkl",
    8917: "./data/model_8917.pkl"
}
DEFAULT_MODEL_PATH = "./data/model_all.pkl"

MODEL_CACHE = {}

def load_model_for_mold_code(mold_code):
    path = SPECIALIZED_MODELS.get(mold_code, DEFAULT_MODEL_PATH)
    if path not in MODEL_CACHE:
        MODEL_CACHE[path] = joblib.load(path)
    return MODEL_CACHE[path]

def predict_one_row(row):
    mold_code = int(row["mold_code"])
    model = load_model_for_mold_code(mold_code)
    x_row = row.to_frame().T
    pred = model.predict(x_row)[0]
    prob = model.predict_proba(x_row)[0, 1]
    return pred, prob


df_all = pd.read_csv(DATA_PATH, low_memory=False)

TRAIN_PATH = "./data/train.csv"

def preload_initial_data(n=1000):
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
    X = train_df.tail(n)

    counter_data = {'total_count': 0, 'fail_count': 0, 'timestamps': [], 'failure_rates': []}
    monitoring_data = []

    now = datetime.now()
    base_time = now - timedelta(seconds=10 * n)

    for i, row in enumerate(X.iterrows()):
        pred, prob = predict_one_row(row[1])
        counter_data['total_count'] += 1
        if pred == 1:
            counter_data['fail_count'] += 1

        timestamp = (base_time + timedelta(seconds=10 * i)).strftime("%Y-%m-%d %H:%M:%S")
        failure_rate = counter_data['fail_count'] / counter_data['total_count']
        counter_data['timestamps'].append(timestamp)
        counter_data['failure_rates'].append(failure_rate)

        sensor_data = row[1][sensor_features].to_dict()
        monitoring_data.append(sensor_data)

    return counter_data, [], monitoring_data


def get_card_color(feature, value, valid_ranges):
    if pd.isna(value):
        return "#d4edda"
    ranges = valid_ranges.get(feature, [])
    for r in ranges:
        if isinstance(r, list) and len(r) == 2 and r[0] <= value <= r[1]:
            return ""
    return "#f8d7da"

def wo_layout():
    return html.Div(children=[
        html.Div(id="sensor-card-container", style={"display": "grid", "gap": "10px", "gridTemplateColumns": "repeat(4, 1fr)"}),
        dcc.Dropdown(
            id="time-range-selector",
            options=[
                {"label": "1분", "value": "1min"},
                {"label": "15분", "value": "15min"},
                {"label": "30분", "value": "30min"},
                {"label": "1시간", "value": "1hour"}
            ],
            value="1min",
            style={"width": "150px", "marginBottom": "10px"}
        ),
        # 1. 실시간 그래프 (상단 전체 너비)
        dcc.Graph(
            id="prob-graph",
            className='glass-card',
            style={"height": "400px", "marginBottom": "20px"}
        ),

        # 2. 아래 카드 영역 (2열 구조)
        html.Div(style={"display": "flex", "gap": "20px"}, children=[
            # 왼쪽 세로 카드 묶음 (불량 여부 + 불량 확률 합친 카드)
            html.Div(className='glass-card', style={
                    "padding": "20px",
                    "borderRadius": "10px",
                    "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
                    "color": "#333333",
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "20px",
                    "flex": "1"
                }, children=[
                    html.Div(id="fault-result"),       # 불량 여부 카드 내용
                    html.Div(id="result-prob-card"),   # 불량 확률 카드 내용
                ]),
            
            # 불량기록 기록 카드
            html.Div(className='glass-card',
                     id="fault-record", 
                     style={
                "flex": "2",
                "border": "1px solid #ccc",
                "padding": "20px",
                "borderRadius": "10px",
                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.05)",
                "maxHeight": "200px",          
                "overflowY": "auto",              
                "position": "relative", 
                "scrollBehavior": "smooth" 
            })
        ]),
    ])


######################################
######################################## 인수인계 페이지
######################################


def analytics_layout():
    return html.Div([
        html.H3("인수인계"),
        dcc.Dropdown(
            id='selected-variable',
            options=[{'label': var, 'value': var} for var in sensor_features],
            value=sensor_features[0],
            style={'width': '300px'}
        ),
        dcc.Graph(id="monitoring-graph", className="card-glass"),
        html.Div(id="result-prob-card", style={"display": "none"}),
        html.Div(id="fault-record", style={"display": "none"}),
        html.Div(id="sensor-card-container", style={"display": "none"}),
        dcc.Graph(id="prob-graph", style={"display": "none"}, className="card-glass"),
        dcc.Dropdown(id="time-range-selector",style={"display": "none"})
    ])


def register_callbacks(app):

    @app.callback(
        Output("counter-store", "data"),
        Output("fault-history-store", "data"),
        Output("realtime-monitoring-store", "data"),   # ✅ 추가
        Input("interval", "n_intervals"),
        State("counter-store", "data"),
        State("fault-history-store", "data"),
        State("realtime-monitoring-store", "data"),   # ✅ 추가
        prevent_initial_call=True
    )
    def run_model_background(n_intervals, counter_data, fault_history, monitoring_data):
        if counter_data is None:
            counter_data = {'total_count': 0, 'fail_count': 0, 'timestamps': [], 'failure_rates': []}
        if fault_history is None:
            fault_history = []
        if monitoring_data is None:
            monitoring_data = []


        current_index = counter_data['total_count']
        if current_index >= len(df_all):
            raise PreventUpdate

        row = df_all.iloc[[current_index]]
        pred, prob = predict_one_row(row.iloc[0])

        counter_data['total_count'] += 1
        if pred == 1:
            counter_data['fail_count'] += 1
            log_msg = f"Index {current_index} - 불량 발생 (확률: {prob:.4f})"
            if log_msg not in fault_history:
                fault_history.append(log_msg)

        now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        failure_rate = counter_data['fail_count'] / counter_data['total_count']
        counter_data['timestamps'].append(now_time)
        counter_data['failure_rates'].append(failure_rate)

        # ✅ 센서 데이터를 모니터링 스토어에 누적 저장
        sensor_data = row[sensor_features].iloc[0].to_dict()
        monitoring_data.append(sensor_data)


        return counter_data, fault_history, monitoring_data

    @app.callback(
        Output("result-prob-card", "children"),
        Output("fault-record", "children"),
        Output("prob-graph", "figure"),
        Output("sensor-card-container", "children"),
        Output("fault-history-store", "data", allow_duplicate=True),

        Input("counter-store", "data"),
        State("fault-history-store", "data"),
        Input({'type': 'delete-fault-btn', 'index': ALL}, 'n_clicks'),
        State("url", "pathname"),
        Input("time-range-selector", "value"),
        State("realtime-monitoring-store", "data"),
        State("valid-ranges-store", "data"),
        prevent_initial_call=True
    )

    def update_visual_and_sensor_cards(counter_data, fault_history, delete_clicks, pathname, time_range, monitoring_data, valid_ranges_by_mold):
        if pathname != "/worker":
            raise PreventUpdate

        if not counter_data or counter_data['total_count'] == 0:
            raise PreventUpdate

        ctx = callback_context
        if ctx.triggered and 'delete-fault-btn' in ctx.triggered[0]['prop_id']:
            prop = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
            idx_to_delete = prop.get("index")
            if fault_history and isinstance(idx_to_delete, int) and 0 <= idx_to_delete < len(fault_history):
                fault_history.pop(idx_to_delete)

        current_index = counter_data['total_count'] - 1
        if current_index < 0 or current_index >= len(df_all):
            raise PreventUpdate

        row = df_all.iloc[[current_index]]
        pred, prob = predict_one_row(row.iloc[0])

        # ===== 결과 카드 =====
        result_card_children = [
            html.Div(f"{'불량' if pred == 1 else '양품'} (Index: {current_index})", style={"fontWeight": "bold", "fontSize": "20px", "marginBottom": "10px"}),
            html.P(f"예측 결과: {'불량품' if pred == 1 else '정상품'}"),
            html.P(f"불량 확률: {prob:.4f}")
        ]

        # ===== 고장 기록 =====
        fault_items = [
            html.Div([
                html.Span(rec, style={"marginRight": "10px"}),
                html.Button("삭제", id={'type': 'delete-fault-btn', 'index': i}, n_clicks=0)
            ], style={"display": "flex", "justifyContent": "space-between"})
            for i, rec in enumerate(fault_history[-20:])
        ]
        fault_display = html.Div([html.H5("불량 기록"), *fault_items])

        # ===== 누적 불량률 그래프 =====
        now = datetime.now()
        time_limit_map = {
            "1min": timedelta(minutes=1),
            "15min": timedelta(minutes=15),
            "30min": timedelta(minutes=30),
            "1hour": timedelta(hours=1)
        }
        limit = time_limit_map.get(time_range, timedelta(minutes=1))
        start_time = now - limit

        timestamps = counter_data['timestamps']
        rates = counter_data['failure_rates']

        try:
            datetime_stamps = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in timestamps]
        except:
            return no_update

        filtered_x = []
        filtered_y = []
        for t, r in zip(datetime_stamps, rates):
            if t >= start_time:
                filtered_x.append(t.strftime("%H:%M:%S"))
                filtered_y.append(r)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_x, y=filtered_y,
            mode='lines+markers', name='최근 불량률',
            line=dict(color='red', width=3),
            marker=dict(color='red', size=10, symbol='circle')
        ))

        tickvals = filtered_x if len(filtered_x) <= 5 else [filtered_x[i] for i in np.linspace(0, len(filtered_x) - 1, 5, dtype=int)]

        fig.update_layout(
            title='선택 시간 범위 누적 불량률',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=30, t=50, b=40),
            font=dict(size=13),
            xaxis_title='시간 (실시간)',
            yaxis_title='누적 불량률',
            yaxis_range=[0, 0.1],
            xaxis=dict(tickmode='array', tickvals=tickvals, tickangle=-45),
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            xaxis_gridcolor='lightgray',
            yaxis_gridcolor='lightgray'
        )

        # ===== 센서 카드 =====
        if not monitoring_data:
            raise PreventUpdate
        sensor_row = monitoring_data[current_index]
        mold_code = str(df_all.iloc[current_index]['mold_code'])
        valid_ranges = valid_ranges_by_mold.get(mold_code, {})

        sensor_cards = []
        for feature in sensor_features:
            value = sensor_row.get(feature)
            display_val = f"값: {value:.2f}" if pd.notna(value) else "값: N/A"
            bg_color = get_card_color(feature, value, valid_ranges)
            card_style = {"backgroundColor": bg_color or "rgba(255,255,255,0.3)", "padding": "1rem", "borderRadius": "10px", "textAlign": "center"}
            sensor_cards.append(html.Div([html.H6(feature), html.Div(display_val)], style=card_style))

        return result_card_children, fault_display, fig, sensor_cards, fault_history


def register_monitoring_callbacks(app):
    @app.callback(
        Output("monitoring-graph", "figure"),
        Input("realtime-monitoring-store", "data"),
        Input("selected-variable", "value"),
        prevent_initial_call=False
    )
    def update_monitoring_graph(monitoring_data, selected_var):
        if not monitoring_data or len(monitoring_data) == 0:
            return go.Figure()

        df = pd.DataFrame(monitoring_data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[selected_var], mode='lines+markers'))
        fig.update_layout(title=f"실시간 {selected_var} 변화", xaxis_title="시점", yaxis_title="값",
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        return fig



__all__ = ["wo_layout", "analytics_layout", "register_callbacks", "register_monitoring_callbacks"]