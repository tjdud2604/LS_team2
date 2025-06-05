from dash import html, dcc
import dash_bootstrap_components as dbc

def layout():
    return dbc.Container([
        html.H2("로그인", className="mb-4"),
        dbc.Input(id="input-username", placeholder="아이디 입력", type="text", className="mb-2"),
        dbc.Input(id="input-password", placeholder="비밀번호 입력", type="password", className="mb-2"),
        dbc.Button("로그인", id="submit-login-button", color="primary", className="mb-3"),
        html.Div(id="result-prob-card", style={"display": "none"}),
        html.Div(id="fault-record", style={"display": "none"}),
        html.Div(id="sensor-card-container", style={"display": "none"}),
        dcc.Graph(id="prob-graph", style={"display": "none"})
    ], style={"marginTop": "100px", "maxWidth": "400px"})
