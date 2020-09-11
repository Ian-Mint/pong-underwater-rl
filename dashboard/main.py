"""
Usage:
`python dashboard.py`
"""
import json
import os.path
import time
from typing import List

import chart_studio.plotly as py
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from dashboard import app
from dashboard.data_loader import *
from dashboard.utils import *

grid_searches = get_grid_searches()
grid_search_params = get_all_grid_search_params()


def get_sliders() -> html.P:
    """
    Return a div of hidden divs with slider text above the slider.
    """
    sliders = []
    for grid_search, param_dict in grid_search_params.items():
        for param, values in param_dict.items():
            prefix = grid_search + param
            s = html.Div([
                html.P(children=[param], id=prefix + '-slider-state'),
                dcc.Slider(
                    id=prefix + '-slider',
                    min=1,
                    max=len(values),
                    value=1,
                    step=1,
                    included=False,
                    marks={n + 1: v for n, v in enumerate(values)}
                ),
            ],
                style={'display': 'none'},
                id=prefix + '-slider-div',
            )
            sliders.append(s)
    return html.P(children=sliders, id='grid-search-sliders')


invisible_grid_search_dropdown_div = html.Div(children=[
    dcc.Dropdown(id='grid-search-params-selector',
                 options=[dict(label='', value='')],
                 style={'display': 'none'})],
    id='grid-search-params-selector-div')

# Create app layout
app.layout = html.Div(
    [
        # empty Div to trigger javascript file for graph resizing
        html.Div(id='output-clientside'),

        # Hidden Divs
        html.Div(id='reward-plot-data', style={'display': 'none'}),
        html.Div(id='step-plot-data', style={'display': 'none'}),
        html.Div(id='dummy1', style={'display': 'none'}),
        html.Div(id='dummy2', style={'display': 'none'}),

        # Header
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "RL for Underwater Communication Experiments",
                                    style={"margin-bottom": "0px"},
                                ),
                            ]
                        )
                    ],
                    className="two-half column",
                    id="title",
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),

        # Reward and steps
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            [
                                "Select experiments:",
                                dcc.Dropdown(
                                    id='experiment-selector',
                                    options=get_experiments(),
                                    multi=True
                                ),
                                dcc.Dropdown(
                                    id='plot-type-selector',
                                    options=[dict(label='line', value='line'), dict(label='scatter', value='scatter')],
                                    multi=False,
                                    value='line'
                                ),
                                html.Br(),
                                html.P("Moving average length:", id='moving-avg-slider-text'),
                                dcc.Slider(
                                    id='moving-avg-slider',
                                    min=1,
                                    max=100,
                                    step=1,
                                    value=10,
                                ),
                                dbc.Button("Edit Rewards", color="primary", className="mr-1", id='edit-reward-button'),
                                dbc.Button("Edit Steps", color="success", className="mr-1", id='edit-steps-button')
                            ]
                        ),
                    ],
                    className='pretty_container three columns',
                    id='training-div',
                ),
                html.Div(
                    [
                        dcc.Graph(id='reward-plot')
                    ],
                    className='pretty_container five columns',
                ),
                html.Div(
                    [
                        dcc.Graph(id='step-plot')
                    ],
                    className='pretty_container five columns',
                ),
            ],
            className='flex-display',
            style={'margin-bottom': '25px'}
        ),

        # Parameters
        html.Div(
            [
                html.Div(
                    [
                        dbc.Table(id='experiment-table'),
                    ],
                    id='params-table-div',
                ),
            ],
            className='flex-display',
            style={'margin-bottom': '25px'}
        ),

        # Grid Search
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            [
                                "Select grid search:",
                                dcc.Dropdown(
                                    id='grid-search-selector',
                                    options=grid_searches,
                                    multi=False
                                ),
                                invisible_grid_search_dropdown_div,
                                html.Br(),
                                get_sliders()
                            ]
                        ),
                    ],
                    className='pretty_container four columns',
                    id='grid-search-selector-div',
                ),
                html.Div(
                    [
                        dcc.Graph(id='grid-search-plot')
                    ],
                    className='pretty_container twelve columns',
                ),
            ],
            className='flex-display',
            style={'margin-bottom': '25px'}
        ),
    ],
    id='mainContainer',
    style={'display': 'flex', 'flex-direction': 'column'},
)


@app.callback(Output('grid-search-sliders', 'children'),
              [Input('grid-search-selector', 'value'),
               Input('grid-search-params-selector', 'value')],
              [State('grid-search-sliders', 'children')],
              prevent_initial_call=False)
def show_grid_search_sliders(grid_search: str, params: List[str], state: List):
    """
    Show the sliders that were selected using the multiple dropdown. Hide the others.

    :param grid_search: name of the grid search
    :param params: list of parameters
    :param state: children of the grid-search-sliders Div
    :return: updated state
    """
    ids = []
    for gs, param_dict in grid_search_params.items():
        for p in param_dict.keys():
            ids.append(gs + p + '-slider-div')

    if params:
        selected_ids = {grid_search + p + '-slider-div' for p in params}
    else:
        selected_ids = set()

    for n, i in enumerate(ids):
        if grid_search and (i not in selected_ids):
            state[n]['props']['style'] = None
        else:
            state[n]['props']['style'] = dict(display='none')
    return state


def assign_slider_text_update_callback(grid_search: str, param: str) -> None:
    """
    Register a callback on the text above categorical sliders. It will then update that text according to the current
    selection.

    :param grid_search: the name of the grid search
    :param param: the name of the parameter the slider adjusts
    """
    prefix = grid_search + param

    def slider_text_update(value: int, marks):
        value = marks[str(value)]
        return [f"{param}: {value}"]

    app.callback(output=Output(prefix + '-slider-state', 'children'),
                 inputs=[Input(prefix + '-slider', 'value')],
                 state=[State(prefix + '-slider', 'marks')],
                 prevent_initial_call=False)(slider_text_update)


# Construct slider inputs and assign callbacks to slider labels
slider_inputs = []
slider_marks = []
for gs, param_dict in grid_search_params.items():
    for p in param_dict.keys():
        assign_slider_text_update_callback(gs, p)
        slider_inputs.append(Input(gs + p + '-slider', 'value'))
        slider_marks.append(State(gs + p + '-slider', 'marks'))


@app.callback(Output('grid-search-params-selector-div', 'children'),
              [Input('grid-search-selector', 'value')], )
def make_grid_search_param_selector(grid_search: str) -> List[dcc.Dropdown]:
    if grid_search:
        options = grid_search_params[grid_search].keys()
        options = [dict(label=o, value=o) for o in options]
        return [dcc.Dropdown(
            id='grid-search-params-selector',
            options=options,
            multi=True
        )]
    else:
        return invisible_grid_search_dropdown_div


@app.callback(
    Output('experiment-table', 'children'),
    [Input('experiment-selector', 'value')]
)
def make_experiment_table(experiments: List[str]) -> List:
    if experiments:
        df = get_parameters_df(experiments)
        table = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)
    else:
        table = dbc.Table.from_dataframe(pd.DataFrame())
    return table


@app.callback(
    Output('moving-avg-slider-text', 'children'),
    [Input('moving-avg-slider', 'value')]
)
def update_moving_avg_slider_text(value: int) -> List:
    return [f"Moving average length: {value}"]


@app.callback(
    Output('reward-plot-data', 'children'),
    [Input('experiment-selector', 'value'),
     Input('moving-avg-slider', 'value'),
     Input('plot-type-selector', 'value')]
)
def make_rewards_plot(experiments: List[str], moving_avg_window: int, plot_type: str) -> str:
    if not experiments:
        fig = get_empty_sunburst("Select an experiment")
    else:
        fig = get_reward_plot(experiments, moving_avg_window, plot_type)
    return fig.to_json()


@app.callback(
    Output('reward-plot', 'figure'),
    [Input('reward-plot-data', 'children')]
)
def display_rewards_plot(json_fig) -> go.Figure:
    return load_figure_from_json(json_fig)


@app.callback(
    Output('dummy1', 'children'),
    [Input('edit-reward-button', 'n_clicks')],
    [State('reward-plot-data', 'children'),
     State('experiment-selector', 'value')],
    prevent_initial_call=True
)
def on_rewards_button_click(n_clicks, json_fig, experiments):
    fig = load_figure_from_json(json_fig)
    py.plot(fig, filename=f'rewards {", ".join(experiments)} {time.strftime("%Y-%m-%d %H.%M.%S")}')
    raise PreventUpdate


@app.callback(
    Output('step-plot-data', 'children'),
    [Input('experiment-selector', 'value'),
     Input('moving-avg-slider', 'value'),
     Input('plot-type-selector', 'value')]
)
def make_steps_plot(experiments: List[str], moving_avg_window: int, plot_type: str) -> str:
    if not experiments:
        fig = get_empty_sunburst("Select an experiment")
    else:
        fig = get_step_plot(experiments, moving_avg_window, plot_type)
    return fig.to_json()


@app.callback(
    Output('step-plot', 'figure'),
    [Input('step-plot-data', 'children')]
)
def display_steps_plot(json_fig) -> go.Figure:
    return load_figure_from_json(json_fig)


@app.callback(
    Output('dummy2', 'children'),
    [Input('edit-steps-button', 'n_clicks')],
    [State('step-plot-data', 'children'),
     State('experiment-selector', 'value')],
    prevent_initial_call=True
)
def on_rewards_button_click(n_clicks, json_fig, experiments):
    fig = load_figure_from_json(json_fig)
    py.plot(fig, filename=f'steps {", ".join(experiments)} {time.strftime("%Y-%m-%d %H.%M.%S")}')
    raise PreventUpdate


def load_figure_from_json(json_fig):
    dict_fig = json.loads(json_fig)
    return go.Figure(data=dict_fig['data'], layout=dict_fig['layout'])


@app.callback(
    Output('grid-search-plot', 'figure'),
    [Input('grid-search-selector', 'value'),
     Input('grid-search-params-selector', 'value')] + slider_inputs,
    slider_marks + [State('grid-search-sliders', 'children')]
)
def make_grid_search_plot(grid_search, axis_params, *args):
    args = list(args)
    state = args.pop()
    slider_values = args[:len(args) // 2]
    slider_value_lookup = args[len(args) // 2:]

    slider_params = dict()
    for v, lookup, s in zip(slider_values, slider_value_lookup, state):
        if s['props']['style'] is None:
            p = s['props']['id'].replace(grid_search, '').split('-')[0]
            slider_params[p] = lookup[str(v)]

    if not grid_search:
        return get_empty_sunburst("Select a grid search")

    if not axis_params:
        if not slider_params:
            return get_empty_sunburst("Select a grid search")
        else:
            return get_empty_sunburst(get_grid_search_results_value(grid_search, **slider_params))

    # Create actual plot
    if len(axis_params) == 1:  # single axis plot
        p = axis_params[0]
        x = [float(v) for v in param_dict[p]]

        y = []
        for v in param_dict[p]:
            slider_params[p] = v
            try:
                y.append(float(get_grid_search_results_value(grid_search, **slider_params)))
            except KeyError:
                y.append(None)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                                 marker=dict(color=y, colorscale='Viridis', size=16)))
        fig.update_layout(xaxis_title=p, yaxis_title="Average Final Reward")
        return fig
    elif len(axis_params) == 2:  # 2-axis plot
        x = [float(v) for v in param_dict[axis_params[0]]]
        y = [float(v) for v in param_dict[axis_params[1]]]
        reward = np.zeros((len(param_dict[axis_params[1]]), len(param_dict[axis_params[0]])))
        for i, vx in enumerate(param_dict[axis_params[0]]):
            slider_params[axis_params[0]] = vx
            for j, vy in enumerate(param_dict[axis_params[1]]):
                slider_params[axis_params[1]] = vy
                try:
                    reward[j, i] = (float(get_grid_search_results_value(grid_search, **slider_params)))
                except KeyError:
                    reward[j, i] = None
        fig = go.Figure()
        fig.add_trace(go.Surface(x=x, y=y, z=reward,
                                 hovertemplate=f"{axis_params[0]}: %{{x}}<br>{axis_params[1]}: %{{y}}<br>Reward: %{{z}}"))
        fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                          highlightcolor="limegreen", project_z=True))
        fig.update_layout(scene=dict(xaxis_title=axis_params[0],
                                     yaxis_title=axis_params[1],
                                     zaxis_title="Average Reward",
                                     camera=dict(
                                        up=dict(x=0, y=0, z=1),
                                        center=dict(x=0, y=0, z=-0.5),
                                        eye=dict(x=1.25, y=1.25, z=1.1)
                                    )),
                          margin=dict(l=0, r=0, b=0, t=0))
        return fig
    elif len(axis_params) > 2:
        return get_empty_sunburst("Select 2 or fewer parameters")


@fig_formatter(t=50)
def get_reward_plot(experiments: List[str], moving_avg_len, plot_type) -> go.Figure:
    df = get_rewards_history_df(experiments, moving_avg_len)
    if plot_type == 'scatter':
        return px.scatter(df, labels=dict(value='reward', index='episode', variable='experiment'), opacity=0.3)
    elif plot_type == 'line':
        return px.line(df, labels=dict(value='reward', index='episode', variable='experiment'))
    else:
        raise ValueError(f"Invalid plot type {plot_type}")


@fig_formatter(t=50)
def get_step_plot(experiments: List[str], moving_avg_len, plot_type) -> go.Figure:
    df = get_steps_history_df(experiments, moving_avg_len)
    if plot_type == 'scatter':
        return px.scatter(df, labels=dict(value='steps', index='episode', variable='experiment'), opacity=0.3)
    elif plot_type == 'line':
        return px.line(df, labels=dict(value='steps', index='episode', variable='experiment'))
    else:
        raise ValueError(f"Invalid plot type {plot_type}")



if __name__ == '__main__':
    # noinspection PyTypeChecker
    app.run_server(debug=False,
                   dev_tools_hot_reload=False,
                   host=os.getenv("HOST", "127.0.0.1"),
                   # host=os.getenv("HOST", "192.168.1.10"),
                   port=os.getenv("PORT", "8050"),
                   )
