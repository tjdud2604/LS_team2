from dash import html, dcc, Output, Input, State, callback_context, no_update
import pandas as pd
import plotly.graph_objs as go
import joblib
import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import ALL

def analytics_layout():
    return html.Div([
        html.H3("인수인계"),
        html.P("여기는 실시간 데이터 분석 페이지입니다."),
    ])