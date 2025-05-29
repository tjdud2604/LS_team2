from dash import html, dcc, Output, Input, State, callback_context, no_update
import pandas as pd
import plotly.graph_objs as go
import joblib
import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import ALL

# === 파일 경로 및 데이터 로드 ===
MODEL_PATH = "./data/model_voting.pkl"
DATA_PATH = "./data/train.csv"
DROP_COLS = ['tryshot_signal', 'heating_furnace']
EXCLUDE_COLS = ['id', 'passorfail', 'timestamp', 'time', 'date', 'name', 'line', 'mold_name']

tuned_vote = joblib.load(MODEL_PATH)
df_all = pd.read_csv(DATA_PATH, low_memory=False)

numeric_cols = df_all.select_dtypes(include=['int64', 'float64']).columns.difference(EXCLUDE_COLS + DROP_COLS).tolist()
categorical_cols = df_all.select_dtypes(include=['object']).columns.difference(EXCLUDE_COLS + DROP_COLS).tolist()

def wo_layout():
    return html.Div(
        className="glass-card",
        style={"marginTop": "20px", "padding": "2rem", "position": "relative", "minHeight": "90vh"},
        children=[
            dcc.Store(id='fault-history-store', data=[]),  # 불량 기록 저장
            html.H2("📡 실시간 센서 데이터", style={"marginBottom": "20px"}, className='shiny-text'),
            dcc.Interval(id="interval", interval=2_000, n_intervals=0),

            # Row Container
            html.Div(
                style={"display": "flex", "gap": "20px"},
                children=[
                    # 왼쪽 카드: 결과 및 확률
                    html.Div(
                        id="result-prob-card",
                        style={
                            "flex": "1",
                            "marginBottom": "10px",
                            "backgroundColor": "rgba(255, 255, 255, 0.3)",
                            "borderRadius": "10px",
                            "padding": "15px",
                            "color": "#000",
                            "boxShadow": "0 4px 10px rgba(0,0,0,0.1)",
                            "minHeight": "100px",
                            "display": "flex",
                            "flexDirection": "column",
                            "justifyContent": "center",
                        },
                        className="glass-card"
                    ),

                    # 중간 카드: 불량 기록
                    html.Div(
                        id="fault-record",
                        style={
                            "flex": "1",
                            "height": "60vh",
                            "overflowY": "auto",
                            "padding": "10px",
                            "backgroundColor": "rgba(255, 255, 255, 0.3)",
                            "borderRadius": "10px",
                            "boxShadow": "0 4px 10px rgba(0,0,0,0.1)",
                        },
                        className='glass-card'
                    ),

                    # 오른쪽 카드: 그래프
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.H5("📈 실시간 양불 확률 그래프"),
                            dcc.Graph(id="prob-graph", style={"height": "60vh"})
                        ]
                    )
                ]
            )
        ]
    )

def register_callbacks(app):
    prob_history = []

    @app.callback(
        Output("result-prob-card", "children"),
        Output("fault-record", "children"),
        Output("prob-graph", "figure"),
        Output("fault-history-store", "data"),
        Input("interval", "n_intervals"),
        Input({'type': 'delete-fault-btn', 'index': ALL}, 'n_clicks'),
        State("fault-history-store", "data"),
        prevent_initial_call=False
    )
    def update_dashboard_or_delete(n_intervals, delete_clicks, fault_history):
        ctx = callback_context
        if fault_history is None:
            fault_history = []

        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # 삭제 버튼 클릭 시 처리
        if trigger_id.startswith("{"):
            triggered_index = eval(trigger_id)['index']
            start_index = max(len(fault_history) - 20, 0)
            actual_index = start_index + triggered_index
            if 0 <= actual_index < len(fault_history):
                fault_history = fault_history.copy()  # 복사해서 수정
                fault_history.pop(actual_index)

            # 삭제시 다른 UI는 변경하지 않고 fault_history만 업데이트
            return no_update, no_update, no_update, fault_history

        # interval 업데이트 시 처리
        elif trigger_id == "interval":
            if n_intervals >= len(df_all):
                return no_update, no_update, no_update, no_update

            try:
                row = df_all.iloc[[n_intervals]].copy()
                row = row.drop(columns=DROP_COLS, errors='ignore')
                row['timestamp'] = pd.to_datetime(row['time'] + " " + row['date'], errors='coerce')

                for col in numeric_cols:
                    row[col] = row[col].fillna(row[col].median())
                for col in categorical_cols:
                    row[col] = row[col].fillna(row[col].mode()[0])

                X_row = pd.DataFrame(row[numeric_cols + categorical_cols].values, columns=numeric_cols + categorical_cols)

                pred = tuned_vote.predict(X_row)[0]
                prob = tuned_vote.predict_proba(X_row)[0, 1]

                # 결과 카드 내용 구성
                result_card_children = [
                    html.Div(
                        f"{'❌ 불량' if pred == 1 else '✅ 양품'} (Index: {n_intervals})",
                        style={"fontWeight": "bold", "fontSize": "20px", "marginBottom": "10px"}
                    ),
                    html.P(f"예측 결과: {'불량품' if pred == 1 else '정상품'}"),
                    html.P(f"🔍 불량 확률: {prob:.4f}")
                ]

                if pred == 1:
                    fault_history = fault_history.copy()
                    fault_history.append(f"Index {n_intervals} - 불량 발생 (확률: {prob:.4f})")

                fault_count = len(fault_history[-20:])

                fault_items = []
                for i, rec in enumerate(fault_history[-20:]):
                    fault_items.append(
                        html.Div(
                            [
                                html.Span(rec, style={"marginRight": "10px"}),
                                html.Button(
                                    "삭제",
                                    id={'type': 'delete-fault-btn', 'index': i},
                                    n_clicks=0,
                                    style={
                                        "backgroundColor": "#dc3545",
                                        "color": "white",
                                        "border": "none",
                                        "borderRadius": "4px",
                                        "padding": "2px 8px",
                                        "cursor": "pointer",
                                    }
                                ),
                            ],
                            style={
                                "marginBottom": "0px",
                                "padding": "4px",
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "backgroundColor": "rgba(255,255,255,0.2)",
                                "borderRadius": "5px"
                            }
                        )
                    )

                fault_display_children = [
                    html.H5(
                        "📋불량 기록" + (" ⚠️불량 5건 이상" if fault_count >= 5 else ""),
                        style={"marginBottom": "10px"}
                    ),
                    *fault_items
                ]

                fault_display_style = {
                    "backgroundColor": "rgba(255, 255, 255, 0.3)",
                    "borderRadius": "10px",
                    "padding": "15px",
                    "color": "#000",
                    "boxShadow": "0 4px 10px rgba(0,0,0,0.1)",
                    "maxHeight": "60vh",
                    "overflowY": "auto",
                    "border": "2px solid red" if fault_count >= 5 else "1px solid rgba(0,0,0,0.1)"
                }

                fault_display = html.Div(
                    fault_display_children,
                    style=fault_display_style,
                    className='glass-card'
                )

                prob_history.append(prob)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=prob_history.copy(),
                    mode="lines+markers",
                    name="불량 확률",
                    line=dict(color="red")
                ))
                fig.update_layout(
                    yaxis=dict(title="불량 확률", range=[0, 1]),
                    xaxis=dict(title="시간 흐름 (Interval)"),
                    margin=dict(t=30, l=30, r=30, b=30)
                )

                return result_card_children, fault_display, fig, fault_history

            except Exception as e:
                print(f"[❌ Error] {e}")
                return no_update, no_update, no_update, no_update

        else:
            return no_update, no_update, no_update, no_update


######################################

def analytics_layout():
    return html.Div([
        html.H3("인수인계"),
        html.P("여기는 실시간 데이터 분석 페이지입니다."),
    ])