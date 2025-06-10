from dash import html, dcc
import dash_bootstrap_components as dbc

def layout():
    return html.Div(
        dbc.Container([
            html.H2("로그인", className="mb-4 text-center"),

            html.P("※ 아이디와 비밀번호를 입력하세요.", className="text-muted text-center", style={"marginBottom": "5px"}),
            html.P(["작업자: 아이디 `worker`, 비밀번호 `worker`" 
                    ,html.Br(), "관리자: 아이디 `admin`, 비밀번호 `admin`"],
                   className="text-muted text-center", style={"fontSize": "12px"}),

            dbc.Input(id="input-username", placeholder="아이디 입력", type="text", className="mb-2"),
            dbc.Input(id="input-password", placeholder="비밀번호 입력", type="password", className="mb-3"),
            dbc.Button("로그인", id="submit-login-button", color="primary", className="w-100"),

            # 숨겨진 컴포넌트들
            html.Div(id="result-prob-card", style={"display": "none"}),
            html.Div(id="fault-record", style={"display": "none"}),
            html.Div(id="sensor-card-container", style={"display": "none"}),
            dcc.Graph(id="prob-graph", style={"display": "none"}),
            dcc.Dropdown(id="time-range-selector", style={"display": "none"})
        ],
        className="glass-card",  # 👉 CSS로 꾸미기 용이하게 className 부여
        style={"maxWidth": "400px", "width": "100%", "marginTop" : "100px"}),

        # 로그인 전체 배경 레이아웃 (중앙 고정, 투명)
        style={
            "position": "fixed",      # ← 사이드바 등과 관계없이 화면 기준 고정
            "top": "50%",
            "left": "50%",
            "transform": "translate(-50%, -50%)",
            "width": "100vw",
            "height": "100vh",
            "backgroundColor": "transparent",  # ← 배경 투명
            "zIndex": 9999  # 맨 위에 위치
        },
        className="glass-bg"  # ✨ glassmorphism 등 적용 가능
    )
