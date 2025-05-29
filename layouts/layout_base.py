# 레이아웃 베이스 파일

import dash_bootstrap_components as dbc
from dash import html
from components.sidebar import side_layout

def navbar():
    return dbc.NavbarSimple(
        children=[
            dbc.Button("로그아웃", id="logout-button", color="danger", size="sm", className="ml-auto")
        ],
        brand="주조데이터 모니터링 시스템",
        brand_href="/",
        color="dark",
        dark=True,
        fixed="top",
    )

def layout_with_container(content):
    return html.Div([
        navbar(),
        html.Div(content, style={"marginTop": "80px", "padding": "2rem"})
    ])

def layout_with_sidebar(role, content):
    return html.Div([
        side_layout(role),  # 사이드바
        html.Div(content, className="main-content")  # 콘텐츠
    ])
