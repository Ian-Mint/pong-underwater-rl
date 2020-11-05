import os.path

from plotly import express as px


__all__ = ['fig_formatter', 'load_markdown_text', 'get_empty_sunburst']


def fig_formatter(**kw):
    """
    Decorator for functions that produce figures. By default, all margins are stripped, but the margin sized can be
    set individually.

    :param t: top margin
    :param l: left margin
    :param r: right margin
    :param b: bottom margin
    :return:
    """
    t = kw.get('t', 0)
    l = kw.get('l', 0)
    r = kw.get('r', 0)
    b = kw.get('b', 0)

    def wrap(func):
        def wrapped(*args, **kwargs):
            fig = func(*args, **kwargs)
            fig.update_layout(margin=dict(t=t, l=l, r=r, b=b),
                              paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              )
            return fig

        return wrapped

    return wrap


def load_markdown_text(file: str) -> str:
    """
    Load markdown text from assets

    :param file: name of markdown file (without extension or path)
    :return: contents of markdown file
    """
    with open(os.path.join(os.path.dirname(__file__), 'assets', f'{file}.md'), 'r') as f:
        md = f.read()
    return md


def get_empty_sunburst(text: str):
    """
    Generates an empty sunburst plot with `text` at its center

    :param text: informational text to display
    :return: `plotly` figure
    """
    return px.sunburst(
        {'x'    : [text],
         'value': [1]},
        path=['x'],
        hover_data=None
    )