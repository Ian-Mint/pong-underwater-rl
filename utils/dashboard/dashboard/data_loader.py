import os
import pickle
import re
from typing import List, Tuple, Dict

import pandas as pd

try:
    from ..dashboard import cache
except ImportError:
    from __init__ import cache

EXP_DIR = 'experiments'
__all__ = ['get_grid_searches_for_dropdown', 'get_experiments_for_dropdown', 'get_experiments_list',
           'get_rewards_history_df', 'get_users_list',
           'get_steps_history_df', 'get_parameters_df', 'get_grid_search_params', 'get_grid_search_experiments',
           'get_all_grid_search_params', 'get_grid_search_results_value', 'get_users_for_dropdown',
           'get_grid_search_list']

# find the root dir
root_dir = '/data'


@cache.memoize()
def load_history(experiment_dir: str) -> List[Tuple[float, int]]:
    """
    Load the pickled history

    :param experiment_dir: The directory of the experiment results
    :return: The training history as a list with an entry per episode (reward, steps)
    """
    assert os.path.exists(experiment_dir), f"{experiment_dir} does not exist"

    file = os.path.join(experiment_dir, 'history.p')
    with open(file, 'rb') as f:
        history = pickle.load(f)
    return history


@cache.memoize()
def get_experiments_list(user) -> List[str]:
    """
    Get all experiments in the experiments directory

    :return: List of experiments
    """
    return sorted(_get_directory_listing(EXP_DIR, user))


@cache.memoize()
def get_grid_search_results_value(user, search: str, **kwargs) -> str:
    experiments, series = get_grid_search_results_series(user, search)

    params = [i.split('.')[0] for i in experiments[0].split('-')]
    param_assignment = dict()
    for p in params:
        param_assignment[p] = kwargs.get(p)

    return series[get_grid_search_results_key(param_assignment)].item()


@cache.memoize()
def get_grid_search_results_series(user, search) -> Tuple[List, pd.Series]:
    experiments = get_grid_search_experiments_list(user, search)
    df = _get_history_df(user, experiments, os.path.join('grid-search', search), 0)
    df = get_moving_average(df, 100)
    series = df.iloc[-1]
    return experiments, series


@cache.memoize()
def get_grid_search_results_key(param_assignment: Dict):
    key = ''
    for p, v in param_assignment.items():
        key += f'{p}.{v}-'
    return key[:-1]


@cache.memoize()
def get_grid_search_experiments_list(user, search: str) -> List[str]:
    """
    Get all experiments in the search directory

    :return: List of experiments
    """
    experiments_root = os.path.join(root_dir, user, 'pong-underwater-rl', 'grid-search', search)
    experiments = os.listdir(experiments_root)
    return sorted(experiments)


@cache.memoize()
def get_grid_search_list(user: str) -> List[str]:
    """
    Get all experiments in the search directory

    :return: List of experiments
    """
    experiments_root = os.path.join(root_dir, 'pong-underwater-rl', 'grid-search')
    experiments = os.listdir(experiments_root)
    return sorted(experiments)


@cache.memoize()
def get_experiments_for_dropdown(user) -> List[Dict]:
    """
    Get all experiments in the experiments directory formatted for use in a plotly dropdown

    :return: List of experiments
    """
    return dash_dropdown_list_from_iterable(_get_directory_listing(EXP_DIR, user))


@cache.memoize()
def get_users_for_dropdown() -> List[Dict]:
    """
    Get all experiments in the experiments directory formatted for use in a plotly dropdown

    :return: List of users
    """
    return dash_dropdown_list_from_iterable(os.listdir('/data'))


@cache.memoize()
def get_users_list():
    """
    Get all users in the root directory

    :return: List of users
    """
    users = os.listdir(root_dir)
    return sorted(users)


@cache.memoize()
def get_all_grid_search_params(user) -> Dict[str, Dict[str, List]]:
    """

    :return: e.g. {'experiment1': {'param1': [1, 2, 3], 'param2': [2, 3]},
                   'experiment2': {'param1': [2, 3, 4], 'param3': [1]}
    """
    result = dict()
    for search in get_grid_searches_for_dropdown(user):
        experiments = get_grid_search_experiments_list(user, search['label'])
        result[search['label']] = get_grid_search_params(experiments)
    return result


@cache.memoize()
def get_grid_search_params(experiments) -> Dict[str, List]:
    """
    Get a dictionary of parameters and values used in the grid search

    :param experiments: experiments in the grid search
    :return: e.g. {'param1': [1, 2, 3],
                   'param2': [1]}
    """
    params = [i.split('.')[0] for i in experiments[0].split('-')]
    result = {p: set() for p in params}
    for ex in experiments:
        for k, v in result.items():
            value = re.findall(rf'(?<={k}\.)[\da-zA-z\.]+(?=-|$)', ex)[0]
            v.add(value)
    return {k: sorted(list(v)) for k, v in result.items()}


@cache.memoize()
def get_grid_searches_for_dropdown(user) -> List[Dict]:
    """
    Get all searches in the grid-searches directory formatted for use in a plotly dropdown

    :return: List of grid searches
    """
    return dash_dropdown_list_from_iterable(_get_directory_listing('grid-search', user))


@cache.memoize()
def get_grid_search_experiments(grid_search: str, user) -> List[str]:
    """
    List of all grid search experiments in a particular search

    :param grid_search: directory name
    :param user: matching the path /data/<user>
    :return: list of experiments
    """
    return _get_directory_listing(os.path.join('grid-search', grid_search), user)


def dash_dropdown_list_from_iterable(iterable):
    value = [{'label': v, 'value': v} for v in iterable]
    return sorted(value, key=lambda x: x['label'])


@cache.memoize()
def _get_directory_listing(directory, user) -> List[str]:
    path = os.path.join(root_dir, user, 'pong-underwater-rl', directory)
    return os.listdir(path)


@cache.memoize()
def get_multi_index_history_df(experiments: List[str]) -> pd.DataFrame:
    """
    example:
                 baseline-1        snell-4        snell-5
              reward   step  reward   step  reward   step
        0      -20.0  430.0   -19.0  207.0   -16.0  590.0
        1      -18.0  322.0   -18.0  343.0   -19.0  361.0
        2      -17.0  423.0   -19.0  348.0   -19.0  514.0
        3      -18.0  414.0   -19.0  255.0   -18.0  538.0
        4      -20.0  364.0   -17.0  240.0   -20.0  407.0

    :param experiments:
    :return:
    """
    hist_dict = {}
    for e in experiments:
        history = load_history(os.path.join(root_dir, 'pong-underwater-rl', EXP_DIR, e))
        rewards = [v[0] for v in history]
        steps = [v[1] for v in history]
        hist_dict[e] = {'reward': rewards, 'step': steps}

    df = pd.DataFrame.from_dict({(i, j): hist_dict[i][j]
                                 for i in hist_dict.keys()
                                 for j in hist_dict[i].keys()},
                                orient='index')
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.transpose()
    return df


@cache.memoize()
def _get_history_df(user, experiments, source, selector: int):
    df = pd.DataFrame()
    for e in experiments:
        history = load_history(os.path.join(root_dir, user, 'pong-underwater-rl', source, e))
        rewards = [v[selector] for v in history]

        temp_df = pd.DataFrame(rewards, columns=[e])
        df = pd.concat([df, temp_df], axis=1)
    return df


def get_moving_average(df: pd.DataFrame, moving_avg_len) -> pd.DataFrame:
    if moving_avg_len <= 1:
        return df
    else:
        for column in df.columns:
            df[column] = df[column].rolling(window=moving_avg_len).mean()
    return df


def get_rewards_history_df(user: str, experiments: List[str], moving_avg_len=1) -> pd.DataFrame:
    """
    Get a dataframe of the reward after each episode for each experiment.

    :param user: The selected user
    :param moving_avg_len:
    :param experiments: List of experiments.
    :return: `pd.DataFrame`
    """
    df = _get_history_df(user, experiments, EXP_DIR, 0)
    return get_moving_average(df, moving_avg_len)


def get_steps_history_df(user: str, experiments: List[str], moving_avg_len=1) -> pd.DataFrame:
    """
    Get a dataframe of the number of steps in each episode for each experiment.

    :param user:
    :param moving_avg_len:
    :param experiments: List of experiments.
    :return: `pd.DataFrame`
    """
    df = _get_history_df(user, experiments, EXP_DIR, 1)
    return get_moving_average(df, moving_avg_len)


@cache.memoize()
def get_parameters_df(user, experiments: List[str]):
    df = pd.DataFrame()
    for e in experiments:
        params_dict = dict(experiment=e)
        with open(os.path.join(root_dir, user, 'pong-underwater-rl', EXP_DIR, e, 'output.log')) as f:
            params_dict.update(_parse_parameters(f.readline()))
        params_df = pd.DataFrame(params_dict, index=[e])

        df = pd.concat([df, params_df], axis=0, join='outer')
    return df


def _parse_parameters(log_line: str) -> dict:
    params = re.findall(r'--([a-z-]+)', log_line)

    # remove params irrelevant to training
    _list_try_remove(params, 'store-dir')
    _list_try_remove(params, 'render')
    _list_try_remove(params, 'checkpont')
    _list_try_remove(params, 'history')

    # remove redundant params
    _list_try_remove(params, 'ps')
    _list_try_remove(params, 'pa')
    _list_try_remove(params, 'pl')
    _list_try_remove(params, 'lr')

    result = dict()
    for p in params:
        matches = re.findall(rf'(?<=--{p}\s)(\S*)(?=\s)', log_line)
        assert len(matches) == 1, f"wrong number of matches for param {p}"
        result[p] = matches.pop()
    return result


def _list_try_remove(l: list, item):
    """
    Removes an item if it exists. Does nothing if `item` is not in list.

    :param l: List to modify in place
    :param item: Item to remove
    """
    try:
        l.remove(item)
    except ValueError:
        pass
