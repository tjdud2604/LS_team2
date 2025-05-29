
from dash import html
import dash_bootstrap_components as dbc

def layout():
    return dbc.Container([
        dbc.NavbarSimple(
            brand="주조데이터 모니터링 시스템 - Admin",
            color="dark",
            dark=True,
            fixed="top"
        ),
        html.Div([
            html.H3("관리자 대시보드"),
            # ... 추가 내용 ...
        ], style={"marginTop": "80px"})
    ])
