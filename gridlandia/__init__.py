import os

import dash
import dash_bootstrap_components as dbc
from flask import Flask, redirect, request
from flask.helpers import get_root_path

__version__ = "0.0.1"


def create_app():
    app = Flask(__name__)
    
    app.config['ENV'] = os.environ.get('ENV','development')
    # app.config.from_object("config")
    if app.config["ENV"] == "development":
        app.config["LOGIN_DISABLED"] = False

    # register dashboard
    from gridlandia.dashapp.callbacks import register_callbacks
    from gridlandia.dashapp.layout import layout

    register_dashapp(app, "gridlandia", None, layout, register_callbacks)

    @app.before_request
    def before_request():
        if not request.is_secure and app.env != "development":
            url = request.url.replace("http://", "https://", 1)
            code = 302  # permanent redirect
            return redirect(url, code=code)

    return app


def register_dashapp(app, title, base_pathname, layout, register_callbacks_fun):
    # Meta tags for viewport responsiveness
    meta_viewport = {
        "name": "viewport",
        "content": "width=device-width, initial-scale=1, shrink-to-fit=no",
    }

    if base_pathname is None:
        url_base_pathname = "/"
    else:
        url_base_pathname = f"/{base_pathname}/"

    my_dashapp = dash.Dash(
        __name__,
        server=app,
        url_base_pathname=url_base_pathname,
        assets_folder=get_root_path(__name__) + f"/static",
        meta_tags=[meta_viewport],
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        external_scripts=[
            "https://cdnjs.cloudflare.com/ajax/libs/chroma-js/2.1.0/chroma.min.js"
        ],
    )

    my_dashapp.css.config.serve_locally = True
    my_dashapp.scripts.config.serve_locally = True

    [
        (
            os.environ.get("USERNAME", "demouser"),
            os.environ.get("USERPASSWORD", "demopass"),
        )
    ]

    # auth = dash_auth.BasicAuth(my_dashapp,VALID_USERS)

    with app.app_context():
        my_dashapp.title = title
        my_dashapp.layout = layout
        if register_callbacks_fun:
            register_callbacks_fun(my_dashapp)
