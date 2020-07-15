import dash
import dash_bootstrap_components as dbc
import flask
from flask_caching import Cache


# Initialize app
app = flask.Flask(__name__)
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/',
                     meta_tags=[{"name": "viewport", "content": "width=device-width"}],
                     external_stylesheets=[dbc.themes.BOOTSTRAP],
                     suppress_callback_exceptions=False)

CACHE_CONFIG = {
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR' : 'cache',
}

cache = Cache()
cache.init_app(dash_app.server, config=CACHE_CONFIG)
