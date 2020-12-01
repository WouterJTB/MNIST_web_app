import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import numpy as np
import json
import tflite_runtime.interpreter as tflite
import io
import re
import plotly.graph_objects as go
from PIL import Image
from cairosvg import svg2png
import helpers

# Build App
external_stylesheets = [dbc.themes.DARKLY]
app = dash.Dash(__name__, title='Digit doodle interpreter', external_stylesheets=[external_stylesheets])
fig = helpers.new_figure()
config = {
    "modeBarButtonsToRemove": [
        "toImage",
        "zoom2d",
        "pan2d",
        "zoomIn2d",
        "zoomOut2d",
        "autoScale2d",
        "resetScale2d",
    ],
}

# Define layout
app.layout = html.Div(
    [
        html.H4("Digit doodle interpreter"),
        dcc.Store(id="data-store", storage_type="session"),
        dcc.Graph(id="graph-pic", figure=fig, config=config),
        dbc.Button("Clear canvas", id="clear-graph"),
        dbc.Button("Save image", id="save-image"),
        dcc.Markdown("Prediction"),
        html.Div(id="data-div"),
        html.Details([
            html.Summary('Contents of image storage'),
            dcc.Markdown(
                id='clientside-figure-json'
            )
        ])
    ],
)

# Load model
interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.callback(
    Output("data-div", "children"),
    Output("data-store", "data"),
    Input("save-image", "n_clicks"),
    State("graph-pic", "relayoutData"),
    prevent_initial_call=True,
)
def save_image(n_clicks, relayout_data):
    try:
        shapes=relayout_data["shapes"]
        images=[]
        output_data=[]
        paths=[]
        for shape in shapes:
            paths.append(shape["path"])
        img=helpers.path2img(paths, size=(28,28), stroke_width=4, viewbox_shift=(10,10),viewbox_margin=(20,20))
        images.append(html.Img(src=img))
        input_data = np.array([helpers.img2array(img)], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = np.argmax(interpreter.get_tensor(output_details[0]['index']))
        output_data.append(prediction)
        return prediction, html.Img(src=img)
    except TypeError:
        return None, "No shape found"

@app.callback(
    Output("graph-pic", "figure"),
    Input("clear-graph", "n_clicks"),
)
def clear_figure(n_clicks):
    return helpers.new_figure()


@app.callback(
    Output('clientside-figure-json', 'children'),
    Input('data-store', 'data')
)
def generated_figure_json(data):
    return '```\n'+json.dumps(data, indent=2)+'\n```'


if __name__ == '__main__':
    app.run_server(debug=True)
