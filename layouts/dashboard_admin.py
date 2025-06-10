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
import plotly.figure_factory as ff

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
    'cast_pressure_is_low' : '주조 압력 (300이하 여부)',
    'biscuit_thickness' : '비스켓 두께',
    'high_section_speed_is_abnormal' : '고속 구간 속도 (이상 여부)',
    'sleeve_temperature_is_outlier' : '슬리브 온도 (이상치 여부)'
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
                    font=dict(size=16, family='Arial Black', color='#333333')
                    ), 
                    xaxis_title='생산일자', 
                    yaxis_title='불량률',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(tickformat="%Y-%m-%d")
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

    slider_label_map = {
        'upper_mold_temp1': '온도 조절 슬라이더',
        'upper_mold_temp2': '온도 조절 슬라이더',
        'lower_mold_temp1': '온도 조절 슬라이더',
        'lower_mold_temp2': '온도 조절 슬라이더',
        'sleeve_temperature': '온도 조절 슬라이더',
        'Coolant_temperature': '온도 조절 슬라이더',
        'cast_pressure': '압력 조절 슬라이더',
        'low_section_speed': '속도 조절 슬라이더'
    } ### 슬라이더 수정

    sensor_cards = [html.Div([
        html.Label(SENSOR_KO_NAME.get(sensor, sensor), style={"fontWeight": "bold", "fontSize": "16px"}),

        # ⬇️ 슬라이더 설명 문구 추가
        html.Div(slider_label_map.get(sensor, ""), style={"marginBottom": "0px", "color": "#666666", "fontSize": "13px","lineHeight": "1.2"}),### 슬라이더 수정

        dcc.Slider(
            id=f"slider-{sensor}",
            min=0, max=1, step=1, value=0, marks={},
            tooltip={"placement": "bottom", "always_visible": True}, className="deeppink-slider"
        ),

        dcc.Graph(id=f"sensor-graph-{sensor}", style={"height": "400px"})
    ], className='glass-card', style={
        "padding": "20px",
        "borderRadius": "10px",
        "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
        "color": "#333333",
        "display": "flex",
        "flexDirection": "column",
        "gap": "15px",
        "flex": "1"
    }) for sensor in SENSORS]

    return html.Div([
        html.Div([
        html.H3("공정 품질 분석 대시보드", style={
            "margin": 0,
            "fontWeight": "bold"
        }),
        html.Span("※ 금형 코드 선택 시, 해당 공정의 주요 변수 영향도 및 품질 상태를 확인할 수 있습니다.", style={
            "fontSize": "14px",
            "color": "#666",
            "marginLeft": "12px"
        })
    ], style={
        "display": "flex",
        "alignItems": "baseline",  # 제목 높이에 맞춤
        "marginBottom": "16px"
    }),
        dcc.Store(id="threshold-per-mold", data=DEFAULT_THRESHOLDS_BY_MOLD),
        # dcc.Store(id="valid-ranges-store", data=initial_valid_ranges),

        html.Div([
            html.Label("금형 코드 선택", style={"fontWeight": "bold"}),
            dcc.Dropdown(id="mold-codes-dropdown",
                         options=[{"label": str(m), "value": str(m)} for m in unique_molds],
                         value=default_mold,
                         style={"width": "300px"})
        ], style={"marginBottom": "20px", "marginBottom": "20px"}),

        # 첫번째 행: 변수 중요도 + 혼동 행렬
        html.Div([
            html.Div([
                html.H4("변수 중요도 (Feature Importance)"),
                dcc.Graph(id='feature-importance-graph',className='glass-card', style={'height': '100%', 'width': '100%', 'flexGrow': 1})
            ], style={'width': '50%', 'display': 'flex', 'flexDirection': 'column',
                    'border': 'none', 'borderRadius': '10px', 'padding': '10px'}),

            html.Div([
                html.H4("혼동 행렬 (Confusion Matrix)"),
                dcc.Graph(id='confusion-matrix-graph', className='glass-card', style={'height': '100%', 'width': '100%', 'flexGrow': 1}),
            ], style={'width': '50%', 'display': 'flex', 'flexDirection': 'column',
                    'border': 'none', 'borderRadius': '10px', 'padding': '10px'})
        ], style={'display': 'flex', 'justify-content': 'space-between', 'gap': '20px'}),

        # 두번째 행: P 관리도
        html.Div([
            # 헤더 영역: P 관리도 + 기간 선택
            html.Div([
                # 제목 + 설명 문구 (왼쪽)
                html.Div([
                    html.H4("P 관리도", style={"margin": 0, "marginRight": "12px"}),
                    html.Span("※ 선택한 기간 내 공정의 불량률 변동 및 이상 여부를 확인할 수 있습니다.", style={
                        "fontSize": "14px",
                        "color": "#666"
                    }),
                ], style={"display": "flex", "alignItems": "center"}),

                # 기간 선택 (오른쪽)
                html.Div([
                    html.Label("기간 선택 : ", style={"fontWeight": "bold", "marginRight": "8px"}),
                    dcc.DatePickerRange(
                        id='date-range',
                        min_date_allowed=min_date,
                        max_date_allowed=max_date,
                        start_date=min_date,
                        end_date=max_date,
                        display_format='YYYY-MM-DD'
                    )
                ], style={"display": "flex", "alignItems": "center"})
            ], style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "marginBottom": "10px"
            }),

            # 그래프 영역
            dcc.Graph(id='p-chart', className='glass-card', style={
                'height': '100%',
                'width': '100%',
                'flexGrow': 1
            }),

            # 데이터 없을 경우 메시지
            html.Div(id='no-data-message', style={
                'color': 'red',
                'fontSize': 20,
                'marginTop': '10px'
            })
        ], style={
            'width': '100%',
            'border': 'none',
            'borderRadius': '10px',
            'padding': '10px',
            'marginTop': '20px',
            "marginBottom": "20px"
        }),


        html.Div([
        html.H3("불량 확률 예측 시뮬레이터", style={
            "margin": 0,
            "fontWeight": "bold"
        }),
        html.Span("※ 주요 변수 값을 조정하면 불량 확률을 예측할 수 있습니다.", style={
            "fontSize": "14px",
            "color": "#666",
            "marginLeft": "12px"
        })
    ], style={
        "display": "flex",
        "alignItems": "baseline",  # 제목 높이에 맞춤
        "marginBottom": "16px"
    }),

        html.Div([
            html.Button("Apply", id="apply-button", n_clicks=0, style={
                "fontSize": "20px",
                "padding": "12px 24px",
                "height": "60px",
                "width": "120px",
                "border": "2px solid black",
                "borderRadius": "8px",
                "cursor": "pointer",
                "marginRight": "20px"
            }),
            html.Div(id="prediction-card-container", style={
                "height": "60px",
                "flexGrow": 1  # 남은 공간 모두 차지
            })
        ], style={
            "display": "flex",
            "alignItems": "center",
            "marginBottom": "30px"
        }),
        
        html.Div([
            html.Label("불량률 기준값 컨트롤러", style={"fontWeight": "bold"}),
            html.Div("← → 슬라이더를 좌우로 움직여 불량률 기준을 조정하세요.",
             style={"fontSize": "12px", "color": "#777", "marginBottom": "6px"}),  
            dcc.Slider(id="threshold-slider", min=0, max=10, step=0.01,marks={0: "0", 3: "3", 5 : "5", 7 : "7" ,10: "10"},
                       tooltip={"placement": "bottom", "always_visible": True}),  
            ], className="glass-card green-slider", style={
            "padding": "20px",
            "borderRadius": "10px",
            "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
            "marginBottom": "30px",
            "color": "#333333",
            "display": "flex",
            "flexDirection": "column",
            "gap": "10px"
            }),

        

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
        Output("prediction-card-container", "children"),
        [Input("mold-codes-dropdown", "value"),
         Input("apply-button", "n_clicks")],
        [State("mold-codes-dropdown", "value"),
         *[State(f"slider-{sensor}", "value") for sensor in SENSORS]]
    )

    def update_prediction_card(mold_code, n_clicks, mold_code_state, *slider_inputs):
        ctx = callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        user_input = dict(zip(SENSORS, slider_inputs))
        filtered = data[data["mold_code"] == int(mold_code)]

        card_style = {
            "height": "100%",   # 부모 Div가 60px이므로 여기에 맞춤
            "width": "100%",    # flexGrow로 채운 공간을 모두 차지
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "padding": "0 30px",
            "borderRadius": "12px",
            "boxShadow": "0px 4px 8px rgba(0,0,0,0.1)",
            "fontSize": "20px"
        }

        # apply 버튼 눌렀을 때만 예측 수행
        if trigger_id == "apply-button" and n_clicks:
            sample_row = filtered.iloc[-1].copy()
            for sensor, value in user_input.items():
                sample_row[sensor] = value

            pred, prob = predict_one_row(sample_row)

            if prob >= 0.6:
                color = "danger"
                message = "※ 불량 가능성이 높습니다. 공정 점검이 필요합니다."
            elif prob >= 0.3:
                color = "warning"
                message = "※ 불량 가능성이 존재합니다. 주의가 필요합니다."
            else:
                color = "success"
                message = "※ 불량 가능성은 낮은 편입니다."
                

            return dbc.Card(
                dbc.CardBody([
                    html.Div([  # ⬅ 상단: 불량 확률 수치
                        html.Div("예측 불량 확률 :", style={
                            "fontWeight": "bold", "fontSize": "20px", "color": "white"}),
                        html.Div(f"{prob * 100:.2f} %", style={
                            "fontSize": "24px", "fontWeight": "bold", "color": "white"})
                    ], style={
                        "display": "flex",
                        "alignItems": "center",
                        "gap": "10px",
                        "width": "100%",
                        "justifyContent": "center",
                        "flexWrap": "wrap"
                    }),

                    html.Div(message, style={  # ⬅ 하단: 메시지 줄바꿈 처리됨
                        "marginTop": "10px",
                        "fontSize": "14px",
                        "color": "grey"
                    })
                ]),
                color=color,
                inverse=True,
                style=card_style
            )

        # 드롭다운 선택이나 초기 페이지 진입 시: 초기화 카드 표시
        return dbc.Card(
            dbc.CardBody(
                html.Div([
                    html.Div("예측 불량 확률 :", style={"fontWeight": "bold", "fontSize": "20px"}),
                    html.Div("값을 설정한 뒤 Apply 버튼을 눌러주세요.", style={"fontSize": "20px", "fontWeight": "bold"})
                ], style={"display": "flex", "alignItems": "center", "gap": "10px", "width": "100%", "justifyContent": "center"})
            ),
            color="secondary",
            inverse=True,
            style=card_style
        )

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
                fig.add_vline(x=vline_x, line=dict(color='deeppink', dash='dot'),  ## 세로선 색깔 변경
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
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14, color='#333333'),
            xaxis_title="중요도", 
            yaxis_title="변수"
        )
        return fig

    @app.callback(
        Output('confusion-matrix-graph', 'figure'),
        Input('mold-codes-dropdown', 'value')
    )
    def update_confusion_matrix(selected_mold_code):
        cm_df = pd.read_csv(f'./data/model_metrics/confusion_mtx_{selected_mold_code}.csv', index_col=0)
        z = cm_df.values
        z_text = [[f"{val} ({val / np.sum(z) * 100:.1f}%)" for val in row] for row in z]

        fig = ff.create_annotated_heatmap(
            z,
            x=["예측 양품", "예측 불량"],
            y=["실제 양품", "실제 불량"],
            annotation_text=z_text,
            colorscale='Blues',
            showscale=True
        )


        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14, color='#333333'),
            xaxis_title="예측 값",
            yaxis_title="실제 값"
        )
        return fig


__all__ = ["ad_layout", "register_callbacks"]    
