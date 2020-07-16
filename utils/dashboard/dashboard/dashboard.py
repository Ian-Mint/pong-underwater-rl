"""
Usage:
`python dashboard.py`
"""
import os
from typing import List, Callable, Dict, Tuple

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate

INTERVAL = 5e3

try:
    from .utils import *
    from .data_loader import *
    from ..dashboard import dash_app, app, cache
except ImportError:
    from utils import *
    from data_loader import *
    from __init__ import dash_app, app, cache


def create_slider(user, grid_search, param, values):
    """
    Return a slider

    :param user: Selected user
    :param grid_search: Selected grid search
    :param param: Selected parameter
    :param values: Parameter values
    :return: `dcc.Slider`
    """
    s = html.Div([
        html.P(children=[param], id=dict(user=user, grid_search=grid_search, param=param, type='grid-slider-state')),
        dcc.Slider(
            id=dict(user=user, grid_search=grid_search, param=param, type='grid-slider'),
            min=1,
            max=len(values),
            value=1,
            step=1,
            included=False,
            marks={n + 1: v for n, v in enumerate(values)}
        ),
    ],
        style={'display': 'none'},
        id=dict(user=user, grid_search=grid_search, param=param, type='grid-slider-div'),
    )
    return s


invisible_grid_search_dropdown_div = html.Div(children=[
    dcc.Dropdown(id='grid-search-params-selector',
                 options=[dict(label='', value='')],
                 style={'display': 'none'})],
    id='grid-search-params-selector-div')

# Create app layout
dash_app.layout = html.Div(
    [
        # empty Div to trigger javascript file for graph resizing
        html.Div(id='output-clientside'),

        # Storage
        dcc.Store(id='grid-searches-store', storage_type='session'),
        dcc.Store(id='grid-search-params-store', storage_type='session'),
        dcc.Store(id='slider-inputs-store', storage_type='session'),
        dcc.Store(id='slider-marks-store', storage_type='session'),

        # Intervals
        dcc.Interval(id='user-monitor', interval=INTERVAL, n_intervals=0),
        dcc.Interval(id='experiments-monitor', interval=INTERVAL, n_intervals=0),
        dcc.Interval(id='grid-search-monitor', interval=INTERVAL, n_intervals=0),
        # todo: dynamically update plot using experiment-history-monitor
        dcc.Interval(id='experiment-history-monitor', interval=INTERVAL, n_intervals=0),

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

        # User selection
        html.Div(
            html.P([
                "Select user:",
                dcc.Dropdown(
                    id='user-selector',
                    options=get_users_for_dropdown(),
                    multi=False
                )
            ],
                id='user-selector-p'
            )
        ),

        # Reward and steps
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            [
                                "Experiments:",
                                dcc.Dropdown(
                                    id='experiment-selector',
                                    options=[dict(label='', value='')],
                                    placeholder='select a user first',
                                    multi=True
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
                                    options=[dict(label='select a user', value='')],
                                    multi=False
                                ),
                                invisible_grid_search_dropdown_div,
                                html.Br(),
                                html.P(children=[], id='grid-search-sliders')
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


@dash_app.callback(Output('user-selector', 'options'),
                   [Input('user-monitor', 'n_intervals')], )
def monitor_experiments_list(n_intervals):
    """
    Monitors the root directory for new users.

    :param n_intervals: the number of times the interval timer has elapsed
    """
    return monitor_memoized(get_users_list, get_users_for_dropdown)


def monitor_memoized(memoized: Callable, f: Callable, *args):
    """
    Monitor memoized function `memoized` for changes with arguments `*args`

    :param memoized: memoized function whose output is monitored
    :param f: another memoized function whose output is returns. Called with `*args`.
    :returns: `f(*args)`
    """
    past = memoized(*args)
    cache.delete_memoized(memoized, *args)
    if past == memoized(*args):
        raise PreventUpdate
    else:
        cache.delete_memoized(f, *args)
        return f(*args)


@dash_app.callback(Output('grid-search-store', 'data'),
                   [Input('user-selector', 'value'), Input('grid-search-monitor', 'n_intervals')], )
def update_grid_search_store(user, n_intervals):
    trigger_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'user-selector':
        return get_grid_searches_for_dropdown(user)
    elif trigger_id == 'grid-search-monitor':
        return monitor_memoized(get_grid_search_list, get_grid_searches_for_dropdown, user)
    else:
        raise ValueError(f"Unexpected trigger id {trigger_id}")


@dash_app.callback(Output('grid-search-selector', 'options'),
                   [Input('grid-search-store', 'modified_timestamp')],
                   [State('grid-search-store', 'data')])
def update_grid_search_selector(ts, data):
    if ts is None:
        raise PreventUpdate
    return data


@dash_app.callback([Output('experiment-selector', 'options'), Output('experiment-selector', 'placeholder')],
                   [Input('experiments-monitor', 'n_intervals'), Input('user-selector', 'value')],
                   [State('experiment-selector', 'placeholder')])
def update_experiment_selector(n_intervals, user, placeholder):
    trigger_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'user-selector':
        return get_experiments_for_dropdown(user), placeholder
    elif trigger_id == 'experiments-monitor':
        return monitor_memoized(get_experiments_list, get_experiments_for_dropdown, user), "Select Experiments"
    else:
        raise ValueError(f"Unexpected trigger id {trigger_id}")


@dash_app.callback(Output('grid-search-sliders', 'children'),
                   [Input('grid-search-params-selector', 'value')],
                   [State('user-selector', 'value'),
                    State('grid-search-selector', 'value'),
                    State('grid-search-sliders', 'children')],
                   prevent_initial_call=False)
def make_grid_search_sliders(params: List[str], user: str, grid_search: str, sliders: List,):
    """
    Show the sliders that were selected using the multiple dropdown. Hide the others.

    :param user: the selected user
    :param grid_search: name of the grid search
    :param params: list of parameters
    :param sliders: children of the grid-search-sliders Div
    :return: updated state
    """
    ids = []
    param_dict = get_all_grid_search_params(user)[grid_search]

    new_sliders_set = {create_slider(user, grid_search, k, v) for k, v in param_dict if k in params}
    sliders_set = set(sliders).union(new_sliders_set)

    return sorted(list(sliders_set), key=None)  # todo: sort by slider id


@dash_app.callback(Output(dict(type='grid-slider-state', user=ALL, grid_search=ALL, param=ALL), 'children'),
                   [Input(dict(type='grid-slider', user=ALL, grid_search=ALL, param=ALL), 'value')],
                   [State(dict(type='grid-slider', user=ALL, grid_search=ALL, param=ALL), 'marks')])
def slider_text_update(sliders: List[int], slider_lookup: List[Dict]) -> Tuple[List[str]]:
    result = []
    for slider_value, sl in zip(sliders, slider_lookup):
        result.append([f"param: {sl[slider_value]}"])

    return tuple(result)


@dash_app.callback(Output('grid-search-params-selector-div', 'children'),
                   [Input('user-selector', 'value'),
                    Input('grid-search-selector', 'value')], )
def make_grid_search_param_selector(user: str, grid_search: str) -> List[dcc.Dropdown]:
    if grid_search:
        options = get_all_grid_search_params(user)[grid_search].keys()
        options = [dict(label=o, value=o) for o in options]
        return [dcc.Dropdown(
            id='grid-search-params-selector',
            options=options,
            multi=True
        )]
    else:
        return invisible_grid_search_dropdown_div


@dash_app.callback(
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


@dash_app.callback(
    Output('moving-avg-slider-text', 'children'),
    [Input('moving-avg-slider', 'value')]
)
def update_moving_avg_slider_text(value: int) -> List:
    return [f"Moving average length: {value}"]


@dash_app.callback(
    Output('reward-plot', 'figure'),
    [Input('experiment-selector', 'value'),
     Input('moving-avg-slider', 'value')]
)
def make_rewards_plot(experiments: List[str], moving_avg_window: int) -> go.Figure:
    if not experiments:
        fig = get_empty_sunburst("Select an experiment")
    else:
        fig = get_reward_plot(experiments, moving_avg_window)
    return fig


@dash_app.callback(
    Output('step-plot', 'figure'),
    [Input('experiment-selector', 'value'),
     Input('moving-avg-slider', 'value')]
)
def make_rewards_plot(experiments: List[str], moving_avg_window: int) -> go.Figure:
    if not experiments:
        fig = get_empty_sunburst("Select an experiment")
    else:
        fig = get_step_plot(experiments, moving_avg_window)
    return fig


@dash_app.callback(
    Output('grid-search-plot', 'figure'),
    [Input('user-selector', 'value'),
     Input('grid-search-selector', 'value'),
     Input('grid-search-params-selector', 'value'),
     Input('slider-inputs-store', 'data')],
    [State('slider-state-store', 'data'),
     State('grid-search-sliders', 'children')]
)
def make_grid_search_plot(user, grid_search, axis_params, slider_values, slider_value_lookup, state):
    """
    Return the grid search plot.

    :param user: The selected user
    :param grid_search: The selected grid search
    :param axis_params: If 1 selected, this is the x-axis of the scatter plot. If 2 selected, these are the x and y
                        axes of the surface plot.
    :param slider_values: values of the sliders
    :param slider_value_lookup: slider markers dict providing a lookup from slider value to real value
    :param state: used to determine which sliders are visible and therefore, active
    :return:
    """
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
            return get_empty_sunburst(get_grid_search_results_value(user, grid_search, **slider_params))

    # Create actual plot
    param_dict = next(get_all_grid_search_params(user).values())
    if len(axis_params) == 1:  # single axis plot
        p = axis_params[0]
        x = [float(v) for v in param_dict[p]]

        y = []
        for v in param_dict[p]:
            slider_params[p] = v
            try:
                y.append(float(get_grid_search_results_value(user, grid_search, **slider_params)))
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
                    reward[j, i] = (float(get_grid_search_results_value(user, grid_search, **slider_params)))
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
def get_reward_plot(experiments: List[str], moving_avg_len) -> go.Figure:
    df = get_rewards_history_df(experiments, moving_avg_len)
    return px.line(df, labels=dict(value='reward', index='episode', variable='experiment'))


@fig_formatter(t=50)
def get_step_plot(experiments: List[str], moving_avg_len) -> go.Figure:
    df = get_steps_history_df(experiments, moving_avg_len)
    return px.line(df, labels=dict(value='steps', index='episode', variable='experiment'))


if __name__ == '__main__':
    # noinspection PyTypeChecker
    # app.run(host='127.0.0.1', debug=True, port=8050)

    dash_app.run_server(debug=True,
                        dev_tools_hot_reload=False,
                        host=os.getenv("HOST", "127.0.0.1"),
                        # host=os.getenv("HOST", "192.168.1.10"),
                        port=os.getenv("PORT", "8050"),
                        )
