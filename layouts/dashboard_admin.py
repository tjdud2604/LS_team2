# dashboard_admin.py

from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import datetime, joblib
from dash.exceptions import PreventUpdate
from dash import callback_context 
import os

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

SENSOR_KO_NAME = {
    'cast_pressure': '주조 압력',
    'lower_mold_temp1': '하부 금형 온도1',
    'lower_mold_temp2': '하부 금형 온도2',
    'upper_mold_temp1': '상부 금형 온도1',
    'upper_mold_temp2': '상부 금형 온도2',
    'facility_operation_cycleTime': '설비 작동 사이클 시간',
    'sleeve_temperature': '슬리브 온도',
    'low_section_speed': '저속 구간 속도',
    'high_section_speed': '고속 구간 속도',
    'Coolant_temperature' : '냉각수 온도',
    'cast_pressure_is_low' : '주조 압력 (300이하)'
}### 한글 추가됨

BINNED_SENSORS = [
    'upper_mold_temp1', 'upper_mold_temp2',
    'lower_mold_temp1', 'lower_mold_temp2',
    'sleeve_temperature'
]

TRAIN_PATH = "./data/train.csv"
IMPORTANCE_PATH = "./data/importance_all.pkl"

# 전역 데이터 불러오기 // 날짜 선택 기능    
data = pd.read_csv(TRAIN_PATH, low_memory=False)
data['time'] = pd.to_datetime(data['time'], errors='coerce')
data['prod_date'] = data['time'].dt.date

min_date = min(data['prod_date'])
max_date = max(data['prod_date'])

# 전역 importance 불러오기 (미리 한 번만 로드)
importance_all = joblib.load(IMPORTANCE_PATH)

# 관리도 생성 로직
def make_p_chart(df_sub, min_n=30):
    summary = df_sub.groupby('prod_date').agg(
        total_count=('passorfail', 'count'),
        fail_count=('passorfail', lambda x: (x == 1).sum())
    ).reset_index()

    summary = summary[summary['total_count'] >= min_n]
    if summary.empty:
        return None

    summary['p_i'] = summary['fail_count'] / summary['total_count']
    p_hat = summary['fail_count'].sum() / summary['total_count'].sum()
    summary['UCL'] = p_hat + 3 * np.sqrt(p_hat * (1 - p_hat) / summary['total_count'])
    summary['LCL'] = (p_hat - 3 * np.sqrt(p_hat * (1 - p_hat) / summary['total_count'])).clip(lower=0)
    summary['CL'] = p_hat

    # 이상 정상 구분
    normal = (summary['p_i'] <= summary['UCL']) & (summary['p_i'] >= summary['LCL'])
    outlier = ~normal

    # Plotly 그래프
    fig = go.Figure()

    # 전체 선
    fig.add_trace(go.Scatter(x=summary['prod_date'], y=summary['p_i'], mode='lines', name='공정 불량률',
                              line=dict(color='cornflowerblue')))

    # 정상 점
    fig.add_trace(go.Scatter(x=summary['prod_date'][normal], y=summary['p_i'][normal], mode='markers',
                              name='관리 기준 내', marker=dict(color='cornflowerblue', size=8)))

    # 이상 점
    fig.add_trace(go.Scatter(x=summary['prod_date'][outlier], y=summary['p_i'][outlier], mode='markers',
                              name='관리기준 초과 (이상 발생)', marker=dict(color='red', size=10, line=dict(color='black', width=1))))

    # 관리한계선
    fig.add_trace(go.Scatter(x=summary['prod_date'], y=summary['UCL'], mode='lines', name='관리 상한 (UCL)',
                              line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=summary['prod_date'], y=summary['LCL'], mode='lines', name='관리 하한 (LCL)',
                              line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=summary['prod_date'], y=summary['CL'], mode='lines', name='중심선 (CL)',
                              line=dict(color='green')))

    fig.update_layout(title=dict(
                    text='p 관리도',
                    font=dict(size=16, family='Arial Black', color='#333333')
                    ), 
                    xaxis_title='생산일자', 
                    yaxis_title='불량률',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)')
    return fig

def compute_slider_ranges_and_modes(df_mold):
        slider_info = {}
        for sensor in SENSORS:
            df_sensor = df_mold[[sensor]].dropna()
            if df_sensor.empty or df_sensor[sensor].nunique() < 2:
                slider_info[sensor] = {"min": 0, "max": 100, "mode": 0}
                continue

            Q1 = df_sensor[sensor].quantile(0.25)
            Q3 = df_sensor[sensor].quantile(0.75)
            IQR = Q3 - Q1
            lower = max(Q1 - 1.5 * IQR, 0)
            upper = Q3 + 1.5 * IQR
            mode_val = df_sensor[sensor].mode().iloc[0]

            slider_info[sensor] = {
                "min": round(lower, 2),
                "max": round(upper, 2),
                "mode": round(mode_val, 2)
            }
        return slider_info

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

def load_model_for_mold_code(mold_code):
    model_path = f"./data/model_{mold_code}.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"모델 파일 없음: {model_path}")

# 1개의 row에 대해 예측 수행 함수
def predict_one_row(row):
    mold_code = int(row["mold_code"])
    model = load_model_for_mold_code(mold_code)
    x_row = pd.DataFrame([row])
    pred = model.predict(x_row)[0]
    prob = model.predict_proba(x_row)[0, 1]
    return pred, prob











def ad_layout():
    global data
    df = data.copy()
    unique_molds = sorted(data['mold_code'].unique())
    default_mold = "8722"

    initial_valid_ranges = compute_valid_ranges(data, DEFAULT_THRESHOLDS_BY_MOLD)

    sensor_cards = [html.Div([
        html.Label(SENSOR_KO_NAME.get(sensor, sensor)),
        dcc.Slider(
            id=f"slider-{sensor}",
            min=0, max=1, step=1, value=0, marks={},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        dcc.Graph(id=f"sensor-graph-{sensor}", style={"height": "400px"})
    ], className='glass-card', style={
        "padding": "20px",
        "borderRadius": "10px",
        "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
        "color": "#333333",
        "display": "flex",
        "flexDirection": "column",
        "gap": "20px",
        "flex": "1"
    })
    for sensor in SENSORS]

    return html.Div([
        html.H3("확인용 - mold_code확인용"),
        dcc.Store(id="threshold-per-mold", data=DEFAULT_THRESHOLDS_BY_MOLD),
        # dcc.Store(id="valid-ranges-store", data=initial_valid_ranges),

        html.Div([
            html.Label("mold_code", style={"fontWeight": "bold"}),
            dcc.Dropdown(id="mold-codes-dropdown",
                         options=[{"label": str(m), "value": str(m)} for m in unique_molds],
                         value=default_mold,
                         style={"width": "300px"})
        ], style={"marginBottom": "20px"}),

        html.Div([
            html.Div([
                # html.H4("변수 중요도 (Feature Importance)"),
                dcc.Graph(id='feature-importance-graph',className='glass-card', style={'height': '100%', 'flexGrow': 1})
            ], style={'width': '48%', 'display': 'flex', 'flexDirection': 'column', 'border': '1px solid lightgray',
                    'borderRadius': '10px', 'padding': '10px'}),

            html.Div([
                html.Div([
                    html.Label("기간 선택:"),
                    dcc.DatePickerRange(
                        id='date-range',
                        min_date_allowed=min_date,
                        max_date_allowed=max_date,
                        start_date=min_date,
                        end_date=max_date,
                        display_format='YYYY-MM-DD'
                    ),
                ], style={"marginBottom": "10px"}),

                dcc.Graph(id='p-chart', style={'height': '450px'}),
                html.Div(id='no-data-message', style={'color': 'red', 'fontSize': 20, 'marginTop': '10px'})
            ],className='glass-card', style={'width': '48%', 'display': 'flex', 'flexDirection': 'column',
                    'border': '1px solid lightgray', 'borderRadius': '10px', 'padding': '10px'})
        ], style={'display': 'flex', 'justify-content': 'space-between', 'gap': '20px', 'align-items': 'stretch'}),


        html.Div([
            html.Label("확인용 (%)", style={"fontWeight": "bold"}),
            dcc.Slider(id="threshold-slider", min=0, max=10, step=0.01,marks={0: "0", 3: "3", 5 : "5", 7 : "7" ,10: "10"},
                       tooltip={"placement": "bottom", "always_visible": True})
        ], style={"marginBottom": "30px"}),

        html.Div(sensor_cards, style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "20px"}),
        html.Div([html.Button("Apply", id="apply-button", n_clicks=0),html.Div(id="prediction-card-container")], style={"marginTop": "20px"}),
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

    @app.callback(  # ✅ 이 부분에 추가
        Output("prediction-card-container", "children"),
        [Input("mold-codes-dropdown", "value"),
         Input("apply-button", "n_clicks")],
        [State("mold-codes-dropdown", "value"),
         *[State(f"slider-{sensor}", "value") for sensor in SENSORS]]
    )

    def update_prediction_card(mold_code, n_clicks, mold_code_state, *slider_inputs):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == "mold-codes-dropdown":
            return ""  # 드롭다운 변경 시 카드 초기화

        if trigger_id == "apply-button":
            if not n_clicks:
                return ""

            user_input = dict(zip(SENSORS, slider_inputs))
            filtered = data[data["mold_code"] == int(mold_code)]

            if filtered.empty:
                return dbc.Alert(f"{mold_code}에 해당하는 데이터가 없습니다.", color="danger")

            sample_row = filtered.iloc[-1].copy()
            for sensor, value in user_input.items():
                sample_row[sensor] = value

            pred, prob = predict_one_row(sample_row)

            if prob >= 0.8:
                color = "danger"
            elif prob >= 0.5:
                color = "warning"
            else:
                color = "success"

            card = dbc.Card(
                dbc.CardBody([
                    html.H4("예측 불량률", className="card-title"),
                    html.H2(f"{prob*100:.2f} %", className="card-text")
                ]),
                color=color,
                inverse=True
            )
            return card

    @app.callback(
        [Output(f"sensor-graph-{sensor}", "figure") for sensor in SENSORS],
        [Input("mold-codes-dropdown", "value"),
        Input("threshold-slider", "value")] +
        [Input(f"slider-{sensor}", "value") for sensor in SENSORS]
    )


    def update_graphs_with_vlines(selected_mold, slider_val, *vline_values):
        global data
        df = data.copy()
        mold_data = df[df['mold_code'] == int(selected_mold)]
        figures = []
        threshold_map = slider_val / 100

        for i, sensor in enumerate(SENSORS):
            vline_x = vline_values[i]
            df_sensor = mold_data[[sensor, 'passorfail']].dropna()
            if df_sensor.empty or df_sensor[sensor].nunique() < 2:
                figures.append(go.Figure())
                continue

            # IQR 기반 이상치 제거
            Q1 = df_sensor[sensor].quantile(0.25)
            Q3 = df_sensor[sensor].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_sensor = df_sensor[(df_sensor[sensor] >= lower) & (df_sensor[sensor] <= upper)]

            if sensor in BINNED_SENSORS:
                df_sensor['sensor_bin'] = (df_sensor[sensor] // 5) * 5  # 5 단위 binning
                grouped = (
                    df_sensor.groupby('sensor_bin')
                    .agg(total=('passorfail', 'count'),
                        fail=('passorfail', lambda x: (x == 1).sum()))
                    .reset_index()
                    .sort_values(by='sensor_bin')
                )
                grouped['fail_rate'] = grouped['fail'] / grouped['total']
                x_vals = grouped['sensor_bin']
            else:
                grouped = (
                    df_sensor.groupby(sensor)
                    .agg(total=('passorfail', 'count'),
                        fail=('passorfail', lambda x: (x == 1).sum()))
                    .reset_index()
                    .sort_values(by=sensor)
                )
                grouped['fail_rate'] = grouped['fail'] / grouped['total']
                x_vals = grouped[sensor]

            # ▶ 그래프 구성
            fig = go.Figure()
            fig.add_trace(go.Bar(x=x_vals, y=grouped['total'], name='총 생산 수', yaxis='y1'))
            fig.add_trace(go.Scatter(x=x_vals, y=grouped['fail_rate'], name='불량률',
                                    mode='lines+markers', yaxis='y2', marker_color='red'))

            # ▶ 기준선
            fig.add_hline(y=threshold_map, yref='y2', line=dict(color='green', dash='dash'),
                        annotation_text=f"기준선: {threshold_map:.2%}", annotation_position="bottom right")

            # ▶ 세로선 추가 (슬라이더 값으로)
            if vline_x is not None:
                fig.add_vline(x=vline_x, line=dict(color='orange', dash='dot'),
                            annotation_text=f"x={vline_x}", annotation_position="top right")

            fig.update_layout(
                # title=f"{SENSOR_KO_NAME.get(sensor, sensor)} 분석 (mold_code={selected_mold})",
                margin=dict(t=30, b=30),
                xaxis=dict(title=SENSOR_KO_NAME.get(sensor, sensor)),
                yaxis=dict(title='총 생산 수'),
                yaxis2=dict(title='불량률', overlaying='y', side='right', range=[0, 0.1]),
                plot_bgcolor='rgba(0,0,0,0)',      # 그래프 영역 배경 투명
                paper_bgcolor='rgba(0,0,0,0)' 
            )
            figures.append(fig)
        return figures

    @app.callback(
        [Output(f"slider-{sensor}", "min") for sensor in SENSORS] +
        [Output(f"slider-{sensor}", "max") for sensor in SENSORS] +
        [Output(f"slider-{sensor}", "value") for sensor in SENSORS] +
        [Output(f"slider-{sensor}", "marks") for sensor in SENSORS],
        Input("mold-codes-dropdown", "value")
    )

    def update_sliders_on_mold_change(selected_mold):
        df_mold = data[data["mold_code"] == int(selected_mold)]
        slider_info = compute_slider_ranges_and_modes(df_mold)

        min_vals, max_vals, mode_vals, marks_vals = [], [], [], []

        for sensor in SENSORS:
            min_val = slider_info[sensor]["min"]
            max_val = slider_info[sensor]["max"]
            mode_val = slider_info[sensor]["mode"]

            min_vals.append(min_val)
            max_vals.append(max_val)
            mode_vals.append(mode_val)

            num_marks = 5
            positions = np.linspace(min_val, max_val, num=num_marks)
            marks = {int(round(pos)): str(int(round(pos))) for pos in positions}
            marks_vals.append(marks)

        return min_vals + max_vals + mode_vals + marks_vals

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
    
    @app.callback(
        Output('p-chart', 'figure'),
        Output('no-data-message', 'children'),
        Input('mold-codes-dropdown', 'value'),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    )
    def update_p_chart(mold_code, start_date, end_date):
        global data
        df = data.copy()
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df['prod_date'] = df['time'].dt.date

        start_date = pd.to_datetime(start_date).date() if start_date else df['prod_date'].min()
        end_date = pd.to_datetime(end_date).date() if end_date else df['prod_date'].max()

        filtered = df[
            (df['mold_code'] == int(mold_code)) &
            (df['prod_date'] >= start_date) &
            (df['prod_date'] <= end_date)
        ]

        fig = make_p_chart(filtered)
        if fig is None:
            return go.Figure(), "※ 선택하신 기간에는 표시할 데이터가 없습니다."
        else:
            return fig, ""
        
    @app.callback(
        Output("feature-importance-graph", "figure"),
        Input("mold-codes-dropdown", "value")
    )
    def update_importance_chart(selected_mold):
        importance_series = importance_all.get(selected_mold)
        if importance_series is None:
            return go.Figure()
        
        # 변수명 클린업 (num__ / cat__ 제거)
        importance_series.index = importance_series.index.str.replace('num__', '', regex=False)
        importance_series.index = importance_series.index.str.replace('cat__', '', regex=False)
        importance_series.rename(index=SENSOR_KO_NAME, inplace=True)  ## 추가됨
        
        # Top 10만 뽑고, 높은 중요도가 위에 나오도록 정렬
        top_n = 10
        importance_series = importance_series.sort_values(ascending=False).head(top_n)[::-1]

        fig = go.Figure(go.Bar(
            x=importance_series.values,
            y=importance_series.index,
            orientation='h'
        ))
        fig.update_layout(
            title=dict(
                text="변수 중요도 (Feature Importance)",
                font=dict(size=16, family='Arial Black', color='#333333')),
            xaxis_title="중요도", yaxis_title="변수",plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
        return fig



__all__ = ["ad_layout", "register_callbacks"]    






