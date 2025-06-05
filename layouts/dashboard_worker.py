# dashboard_worker.py

from dash import html, dcc, Output, Input, State, callback_context, no_update
import pandas as pd
import plotly.graph_objs as go
import joblib
import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import ALL
from datetime import datetime
import os
import json
import sys
import numpy as np # numpy import 추가


MODEL_PATH = "./data/model_voting1.pkl"
DATA_PATH = "./data/train.csv"
EXCLUDE_COLS = ['id', 'line', 'name', 'mold_name', 'time', 'date', 'count', 'working','molten_volume','upper_mold_temp3','lower_mold_temp3','registration_time','tryshot_signal','passorfail']

sensor_features = [
    "cast_pressure", "lower_mold_temp2", "low_section_speed", "upper_mold_temp1",
    "upper_mold_temp2", "sleeve_temperature", "lower_mold_temp1", "high_section_speed"
]
sensor_thresholds = {
    "upper_mold_temp1": [(99,104),(154, 999)],
    "upper_mold_temp2": (154, 999),
    "lower_mold_temp1": [(124,134),(159,309)],
    "lower_mold_temp2": [(139,149),(169,999)],
    "low_section_speed": [105, 110, 115],
    "high_section_speed": [(101.5, 102.5),(109,113),(117,118)],
    "cast_pressure": [(311,315),(318,334)],
    "sleeve_temperature": [(160, 210), (460, 530)]
}

tuned_vote = joblib.load(MODEL_PATH)
df_all = pd.read_csv(DATA_PATH, low_memory=False)

numeric_cols = df_all.select_dtypes(include=['int64', 'float64']).columns.difference(EXCLUDE_COLS).tolist()
categorical_cols = df_all.select_dtypes(include=['object']).columns.difference(EXCLUDE_COLS).tolist()

def get_card_color(feature, value):
    if pd.isna(value):
        return "#d4edda"
    threshold = sensor_thresholds.get(feature)
    if isinstance(threshold, tuple):
        return "" if threshold[0] <= value <= threshold[1] else "#f8d7da"
    elif isinstance(threshold, list):
        if all(isinstance(r, tuple) for r in threshold):
            for r in threshold:
                if r[0] <= value <= r[1]:
                    return ""
            return "#f8d7da"
        else:
            return "" if value in threshold else "#f8d7da"
    return ""

def wo_layout():
    return html.Div(children=[
        # 센서 카드 컨테이너 - 항상 제일 위에
        html.Div(id="sensor-card-container", style={"display": "grid", "gap": "10px", "gridTemplateColumns": "repeat(4, 1fr)"}),

        # 1. 그래프 (상단 전체 너비)
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
            
            # 오른쪽 고장 기록 카드
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
        ])
    ])





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
        dcc.Graph(id="prob-graph", style={"display": "none"}, className="card-glass")
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
        X_row = pd.DataFrame(row[numeric_cols + categorical_cols].values, columns=numeric_cols + categorical_cols)
        pred = tuned_vote.predict(X_row)[0]
        prob = tuned_vote.predict_proba(X_row)[0, 1]

        counter_data['total_count'] += 1
        if pred == 1:
            counter_data['fail_count'] += 1
            log_msg = f"Index {current_index} - 불량 발생 (확률: {prob:.4f})"
            if log_msg not in fault_history:
                fault_history.append(log_msg)

        now_time = datetime.now().strftime("%H:%M:%S")
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
        prevent_initial_call=True
    )

    def update_visual(counter_data, fault_history, delete_clicks, pathname):
        if pathname != "/worker":
            raise PreventUpdate
        
        if counter_data is None or counter_data['total_count'] == 0:
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
        X_row = pd.DataFrame(row[numeric_cols + categorical_cols].values, columns=numeric_cols + categorical_cols)
        pred = tuned_vote.predict(X_row)[0]
        prob = tuned_vote.predict_proba(X_row)[0, 1]

        result_card_children = [
            html.Div(f"{'불량' if pred == 1 else '양품'} (Index: {current_index})", style={"fontWeight": "bold", "fontSize": "20px", "marginBottom": "10px"}),
            html.P(f"예측 결과: {'불량품' if pred == 1 else '정상품'}"),
            html.P(f"불량 확률: {prob:.4f}")
        ]

        fault_items = [
            html.Div([
                html.Span(rec, style={"marginRight": "10px"}),
                html.Button("삭제", id={'type': 'delete-fault-btn', 'index': i}, n_clicks=0)
            ], style={"display": "flex", "justifyContent": "space-between"})
            for i, rec in enumerate(fault_history[-20:])
        ]
        fault_display = html.Div([
            html.H5("불량 기록"),
            *fault_items
        ])

        recent_times = counter_data['timestamps'][-60:]
        recent_rates = counter_data['failure_rates'][-60:]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent_times, y=recent_rates, mode='lines+markers', name='최근 불량률', line=dict(color='red', width=3), marker=dict(color='red', size=10, symbol='circle')))
        fig.update_layout(title='최근 60개 예측 기준 누적 불량률', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                          margin=dict(l=50, r=30, t=50, b=40), font=dict(size=13), xaxis_title='시간 (실시간)', 
                          yaxis_title='누적 불량률', yaxis_range=[0, 0.1], xaxis_showgrid=True, yaxis_showgrid=True, 
                          xaxis_gridcolor='lightgray', yaxis_gridcolor='lightgray')

        sensor_cards = []
        for col in sensor_features:
            val = row[col].values[0] if col in row.columns else None
            display_val = f"값: {val:.2f}" if pd.notna(val) else "값: N/A"
            bg_color = get_card_color(col, val)
            card_style = {"backgroundColor": bg_color if bg_color else "rgba(255, 255, 255, 0.3)", "padding": "1rem", "borderRadius": "10px", "boxShadow": "0 2px 6px rgba(0,0,0,0.1)", "textAlign": "center"}
            sensor_cards.append(html.Div(className="sensor-card glass-card", style=card_style, children=[html.H6(col, style={"marginBottom": "10px"}), html.Div(display_val, style={"fontSize": "1.5rem", "fontWeight": "bold"})]))

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