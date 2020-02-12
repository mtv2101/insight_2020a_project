# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Output, Input, State
import pandas as pd
import flask
import numpy as np


from predict_and_plot import get_class_frequency

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_pickle('/Workplace_barometer/output/regex_scores_20200206-221204.pkl')
df = df.dropna(axis=0)
classes, class_counts, uncat_count = get_class_frequency(df)
class_counts = [count/float(len(df)) for count in class_counts]
dropdown_dict = [{'label':c, 'value':c} for c in classes]# list of single-key dicts, one for each drowpdown item



LEFT_COLUMN = dbc.Jumbotron(
    [
        #html.H4(children="Survey Text by Topic", className="display-5"),
        #html.Hr(className="my-2"),
        html.Label("Select a topic", style={"marginTop": 0}, className="lead"),
        dcc.Dropdown(options=dropdown_dict, id='choose_cat'),
        dcc.Markdown(id='comment_text', style={"marginTop": 20}),
    ])



NAVBAR = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(
                        dbc.NavbarBrand("The Workplace Barometer", className="ml-2")
                    ),
                ],
                align="center",
                no_gutters=True,
            ),
            href="https://github.com/mtv2101/insight_2020a_project",
        )
    ],
    color="dark",
    dark=True,
    sticky="top",
)

Survey_Response_Topics = [
    dbc.CardHeader(html.H5("Survey Response Topics")),
    dbc.CardBody(
        [
            dcc.Graph(
                id='Class Frequency',
                figure={
                    'data': [
                        dict(
                            name='Class Frequency',
                            x=classes,
                            y=class_counts,
                            type='bar'
                        )
                    ],
                    'layout': dict(
                        yaxis={'title': '% Survey Responses'},
                        margin={'l': 80, 'b': 80, 't': 30, 'r': 60},
                        hovermode='closest',
                    )
                }
            ),
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

BODY = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(LEFT_COLUMN, md=4, align="center"),
                dbc.Col(dbc.Card(Survey_Response_Topics), md=8),
            ],
            style={"marginTop": 30},
        ),
    ],
    className="mt-12",
)

server = flask.Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], server=server)
app.layout = html.Div(children=[NAVBAR, BODY])


def return_class(class_list, target_class):
    if target_class in class_list:
        return True
    else:
        return False


@app.callback(
    Output("comment_text", "children"),
    [Input("choose_cat", "value")],
)
def get_comment_text(input_value):
    if input_value:
        mask = df.labels.apply(lambda x: input_value in x)
        class_df = df[mask]
        rand_idx = np.random.choice(np.arange(0, len(class_df)), 1)[0]
        comment_text = class_df['text'].iloc[rand_idx]
        return(comment_text)


if __name__ == '__main__':
    app.run_server(debug=True)