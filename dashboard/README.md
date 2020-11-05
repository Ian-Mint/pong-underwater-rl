# Dashboard

To use the dashboard, install `dash` in a `conda` environment
```shell script
conda install dash
```

Modify the last lines of `dashboard.py` as follows to serve the dashboard on [localhost](http:\\localhost:8050)
```python
if __name__ == '__main__':
    # noinspection PyTypeChecker
    app.run_server(debug=False,
                   dev_tools_hot_reload=False,
                   host=os.getenv("HOST", "127.0.0.1"),
                   # host=os.getenv("HOST", "192.168.1.10"),
                   port=os.getenv("PORT", "8050"),
                   )
```

Run the dashboard
```shell script
python dashboard.py
```
