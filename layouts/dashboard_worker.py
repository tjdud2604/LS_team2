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
import numpy as np
from datetime import datetime, timedelta
from feature_engineer import FeatureEngineer
import pandas as pd
import shap
pd.set_option('future.no_silent_downcasting', True)


DATA_PATH = "./data/answer.csv"

sensor_features = [
    "upper_mold_temp1", "upper_mold_temp2", "lower_mold_temp1", "lower_mold_temp2",
    "cast_pressure", "sleeve_temperature", "low_section_speed", "high_section_speed"
]

SPECIALIZED_MODELS = {
    8412: "./data/model_8412.pkl",
    8573: "./data/model_8573.pkl",
    8600: "./data/model_8600.pkl",
    8722: "./data/model_8722.pkl",
    8917: "./data/model_8917.pkl"
}
DEFAULT_MODEL_PATH = "./data/model_all.pkl"

MODEL_CACHE = {}
FAILURE_CAUSE_STATS = {}
EXPLAINER_CACHE = {}

def load_model_for_mold_code(mold_code):
    path = SPECIALIZED_MODELS.get(mold_code, DEFAULT_MODEL_PATH)
    if path not in MODEL_CACHE:
        MODEL_CACHE[path] = joblib.load(path)
    return MODEL_CACHE[path]

def get_explainer_for_model(model, background=None):
    key = id(model)
    if key not in EXPLAINER_CACHE:
        if background is not None:
            EXPLAINER_CACHE[key] = shap.TreeExplainer(
                model,
                data=background,
                feature_perturbation="interventional"
            )
        else:
            EXPLAINER_CACHE[key] = shap.TreeExplainer(
                model,
                feature_perturbation="auto"
            )
    return EXPLAINER_CACHE[key]

def predict_with_shap(row):
    sensor_features_for_shap = [
        'num__upper_mold_temp1', 'num__upper_mold_temp2', 'num__lower_mold_temp1', 'num__lower_mold_temp2',
        'num__cast_pressure', 'num__sleeve_temperature', 'num__low_section_speed', 'num__Coolant_temperature'
    ]
    
    mold_code = int(row["mold_code"])
    pipeline = load_model_for_mold_code(mold_code)
    x_row_raw = row.to_frame().T

    # 전처리
    fe = pipeline.named_steps['feature_engineering']
    x_fe = fe.transform(x_row_raw)
    preprocessor = pipeline.named_steps['preprocessing']
    x_processed = preprocessor.transform(x_fe)
    model = pipeline.named_steps['clf']

    # SHAP 계산
    explainer = get_explainer_for_model(model)
    shap_values = explainer.shap_values(x_processed, check_additivity=False)

    # Binary classification 분기
    if isinstance(shap_values, list):
        shap_for_class = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_for_class = shap_values

    pred = model.predict(x_processed)[0]
    prob = model.predict_proba(x_processed)[0, 1]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = [f"feature_{i}" for i in range(x_processed.shape[1])]

    shap_array = np.array(shap_for_class[0])
    if shap_array.ndim > 1:
        shap_array = shap_array.flatten()

    abs_sum = np.sum(np.abs(shap_array))
    top_features = []

    if abs_sum != 0:
        contrib_list = [(name, abs(val) / abs_sum) for name, val in zip(feature_names, shap_array)]
        filtered = [(name, percent) for name, percent in contrib_list if name in sensor_features_for_shap]

        if filtered:
            # 상위 1개만
            top_features = [name for name, _ in sorted(filtered, key=lambda x: x[1], reverse=True)[:1]]

    return pred, prob, top_features

def update_failure_cause_stats(mold_code, top_features):
    mold_code = str(mold_code)  # 🔄 str 통일
    if mold_code not in FAILURE_CAUSE_STATS:
        FAILURE_CAUSE_STATS[mold_code] = {}

    for f in top_features:
        FAILURE_CAUSE_STATS[mold_code][f] = FAILURE_CAUSE_STATS[mold_code].get(f, 0) + 1

def predict_one_row(row):
    mold_code = int(row["mold_code"])
    model = load_model_for_mold_code(mold_code)
    x_row = row.to_frame().T
    pred = model.predict(x_row)[0]
    prob = model.predict_proba(x_row)[0, 1]
    return pred, prob


df_all = pd.read_csv(DATA_PATH, low_memory=False)

TRAIN_PATH = "./data/sampled_train.csv" 

def preload_initial_data(n=1000):
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
    X = train_df.tail(n)

    counter_data = {'total_count': 0, 'fail_count': 0, 'timestamps': [], 'failure_rates': []}
    production_monitoring = {
        'timestamps': [],
        'mold_codes': [],
        'sensor_data': []
    }
    monitoring_data = []

    # ✅ 통합된 몰드 통계 및 SHAP 요약
    mold_stats = {}

    now = datetime.now()
    base_time = now - timedelta(seconds=10 * n)
    
    for i, row in enumerate(X.iterrows()):
        row_data = row[1]
        pred, prob, shap_summary = predict_with_shap(row_data)
        mold_code = int(row_data['mold_code'])

        # ✅ 몰드코드 초기화 및 생산 수 증가
        if mold_code not in mold_stats:
            mold_stats[mold_code] = {
                'total': 0,
                'fail': 0,
                'shap_summary': {}
            }
        mold_stats[mold_code]['total'] += 1

        counter_data['total_count'] += 1
        timestamp = (base_time + timedelta(seconds=10 * i)).strftime("%Y-%m-%d %H:%M:%S")

        if pred == 1:
            counter_data['fail_count'] += 1
            mold_stats[mold_code]['fail'] += 1

            # ✅ SHAP 피처별 등장 횟수 누적
            for feature in shap_summary:
                shap_counter = mold_stats[mold_code]['shap_summary']
                shap_counter[feature] = shap_counter.get(feature, 0) + 1

        failure_rate = counter_data['fail_count'] / counter_data['total_count']
        counter_data['timestamps'].append(timestamp)
        counter_data['failure_rates'].append(failure_rate)

        sensor_data = row_data[sensor_features].to_dict()
        monitoring_data.append(sensor_data)

        production_monitoring['timestamps'].append(timestamp)
        production_monitoring['mold_codes'].append(mold_code)
        production_monitoring['sensor_data'].append(sensor_data)

    # ✅ 최종 통합된 데이터 반환
    return counter_data, [], monitoring_data, production_monitoring, mold_stats

#################################
# 작업자 1페이지
##################################

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
        dcc.Graph(
            id="prob-graph",
            className='glass-card',
            style={"height": "400px", "marginBottom": "20px"}
        ),

        html.Div(style={"display": "flex", "gap": "20px"}, children=[
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
                    html.Div(id="fault-result"),
                    html.Div(id="result-prob-card"),
                ]),
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


##############################
##############################

mold_codes = list(SPECIALIZED_MODELS.keys())

def analytics_layout():
    return html.Div([
        html.H4("몰드코드별 시간대별 생산 이력", className="glass-card"),
        dcc.Graph(id='mold-time-graph', className="glass-card"),

        dcc.Dropdown(
            id='mold-code-selector',
            options=[{'label': str(code), 'value': str(code)} for code in mold_codes],
            value=str(mold_codes[0]),
            style={'width': '300px', 'marginBottom': '20px'}
        ),

        dcc.Graph(id='mold-failure-bar'),
        dcc.Graph(id='shap-pie-chart'),

        dcc.Dropdown(
            id='selected-variable',
            options=[{'label': var, 'value': var} for var in sensor_features],
            value=[sensor_features[0]],
            multi=True,
            style={'width': '300px'}
        ),

        dcc.Graph(id="monitoring-graph", className="glass-card"),

        # 오류방지용 작성
        html.Div(id="result-prob-card", style={"display": "none"}),
        html.Div(id="fault-record", style={"display": "none"}),
        html.Div(id="sensor-card-container", style={"display": "none"}),
        dcc.Graph(id="prob-graph", style={"display": "none"}, className="card-glass"),
        dcc.Dropdown(id="time-range-selector", style={"display": "none"})
    ])

def register_mold_callbacks(app):
    @app.callback(
        Output("mold-time-graph", "figure"),
        Output("mold-failure-bar", "figure"),
        Output("shap-pie-chart", "figure"),
        Input("production-monitoring-store", "data"),
        Input("mold-code-selector", "value"),
        Input("mold-stats-store", "data"),
        prevent_initial_call=False
    )
    def update_mold_time_graph(production_monitoring, selected_code, mold_stats):
        if selected_code is None or mold_stats is None:
            print("인수인계 문제 발생1")
            raise PreventUpdate
        if not production_monitoring or not production_monitoring.get('timestamps'):
            return go.Figure()

        # production_monitoring → dict → DataFrame 변환
        df_filtered = pd.DataFrame({
            'timestamp': production_monitoring['timestamps'],
            'mold_code': production_monitoring['mold_codes']
        })

        df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])
        df_filtered['mold_code_str'] = df_filtered['mold_code'].astype(str)

        time_fig = go.Figure()

        for code in mold_codes:
            code_str = str(code)
            df_code = df_filtered[df_filtered['mold_code_str'] == code_str]
            time_fig.add_trace(go.Scatter(
                x=df_code['timestamp'],
                y=[code_str] * len(df_code),
                mode='markers',
                marker=dict(size=10),
                name=f'Mold {code_str}'
            ))

        time_fig.update_layout(
            title="몰드코드별 시간대별 생산 이력",
            xaxis_title="시간",
            yaxis_title="Mold Code",
            yaxis=dict(categoryorder='category ascending'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        # === 2. 불량률 막대 그래프 ===
        codes = []
        failure_rates = []
        for code, stats in mold_stats.items():
            try:
                code_int = int(code)  # 보장
            except:
                continue
            total = stats.get("total", 0)
            fail = stats.get("fail", 0)
            if total > 0:
                codes.append(str(code_int))
                failure_rates.append(fail / total)

        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(x=codes, y=failure_rates, name="불량률"))
        bar_fig.update_layout(
            title="몰드코드별 불량률",
            xaxis_title="몰드코드",
            yaxis_title="불량률",
            yaxis_range=[0, 1],
            plot_bgcolor='rgba(0,0,0,0)'
        )

        # === 3. SHAP 피처 파이차트 ===
        selected_stats = mold_stats.get(str(selected_code)) 
        if not selected_stats:
            pie_fig = go.Figure()
            pie_fig.update_layout(title="SHAP 데이터 없음")
            return time_fig, bar_fig, pie_fig

        shap_dict = selected_stats.get("shap_summary", {})
        if shap_dict:
            pie_fig = go.Figure(data=[
                go.Pie(labels=list(shap_dict.keys()), values=list(shap_dict.values()), hole=0.3)
            ])
            pie_fig.update_layout(title=f"{selected_code}번 몰드 SHAP 주요 원인 분포")
        else:
            pie_fig = go.Figure()
            pie_fig.update_layout(title="SHAP 데이터 없음")

        return time_fig, bar_fig, pie_fig



####################################
####################################

def register_callbacks(app):

    @app.callback(
        Output("counter-store", "data"),
        Output("fault-history-store", "data"),
        Output("realtime-monitoring-store", "data"),  
        Output("production-monitoring-store", "data"),
        Output("mold-stats-store", "data"),

        Input("interval", "n_intervals"),

        State("counter-store", "data"),
        State("fault-history-store", "data"),
        State("realtime-monitoring-store", "data"), 
        State("production-monitoring-store", "data"),
        State("mold-stats-store", "data"), 

        prevent_initial_call=True
    )

    def run_model_background(n_intervals, counter_data, fault_history, monitoring_data, production_monitoring, mold_stat):
        if counter_data is None:
            counter_data = {'total_count': 0, 'fail_count': 0, 'timestamps': [], 'failure_rates': []}
        if fault_history is None:
            fault_history = []
        if monitoring_data is None:
            monitoring_data = []
        if production_monitoring is None:
            production_monitoring = {
                'timestamps': [],
                'mold_codes': [],
                'sensor_data': []
        }
            
        if mold_stat is None:
            print("문제발생")
            mold_stat = {}

        current_index = counter_data['total_count']
        if current_index >= len(df_all):
            raise PreventUpdate

        row = df_all.iloc[[current_index]]
        pred, prob, shap_summary = predict_with_shap(row.iloc[0])
        mold_code = str(row.iloc[0]["mold_code"])

        if mold_code not in mold_stat:
            print(f"문제발생: {mold_code}")
            mold_stat[mold_code] = {
                'total': 0,
                'fail': 0,
                'shap_summary': {}
            }
        mold_stat[mold_code]['total'] += 1

        counter_data['total_count'] += 1
        if pred == 1:
            counter_data['fail_count'] += 1
            mold_stat[mold_code]['fail'] += 1

            for feature in shap_summary:
                shap_dict = mold_stat[mold_code]['shap_summary']
                shap_dict[feature] = shap_dict.get(feature, 0) + 1

            update_failure_cause_stats(mold_code, shap_summary)  # 여기서 shap_summary는 실제로 top_features
            top_features = shap_summary  # 이름만 더 명확하게 해주는 용도
            log_msg = f"Index {current_index} - 불량 발생 - 주요 원인: {', '.join(top_features)}"

            if log_msg not in fault_history:
                fault_history.append(log_msg)

        now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        failure_rate = counter_data['fail_count'] / counter_data['total_count']
        counter_data['timestamps'].append(now_time)
        counter_data['failure_rates'].append(failure_rate)

        # ✅ 센서 데이터를 모니터링 스토어에 누적 저장
        sensor_data = row[sensor_features].iloc[0].to_dict()
        monitoring_data.append(sensor_data)

        production_monitoring['timestamps'].append(now_time)
        production_monitoring['mold_codes'].append(mold_code)
        production_monitoring['sensor_data'].append(sensor_data)


        return counter_data, fault_history, monitoring_data, production_monitoring, mold_stat

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

#############################
#################################

def register_monitoring_callbacks(app):
    @app.callback(
        Output("monitoring-graph", "figure"),
        Input("realtime-monitoring-store", "data"),
        Input("selected-variable", "value"),
        Input("mold-code-selector", "value"),
        State("production-monitoring-store", "data"),
        prevent_initial_call=False
    )
    def update_monitoring_graph(monitoring_data, selected_var, selected_code, production_monitoring):
        if not monitoring_data or len(monitoring_data) == 0:
            return go.Figure()

        df = pd.DataFrame(monitoring_data)

        timestamps = production_monitoring['timestamps']
        mold_codes_all = production_monitoring['mold_codes']

        df['timestamp'] = pd.to_datetime(timestamps)
        df['mold_code'] = mold_codes_all

        df['mold_code'] = df['mold_code'].astype(str)
        selected_code = str(selected_code)

        df_filtered = df[df['mold_code'] == selected_code]
        
        fig = go.Figure()

        for var in selected_var:
            fig.add_trace(go.Scatter(
                x=df_filtered['timestamp'],
                y=df_filtered[var],
                mode='lines+markers',
                name=f"{var}"
            ))

        fig.update_layout(
            title=f"전체 몰드코드 - {selected_var} 실시간 변화",
            xaxis_title="시간",
            yaxis_title=", ".join(selected_var),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

################################
####################################

__all__ = ["wo_layout", "analytics_layout", "register_callbacks", "register_monitoring_callbacks", "register_mold_time_callbacks"]