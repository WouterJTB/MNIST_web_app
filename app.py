import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import numpy as np
import tflite_runtime.interpreter as tflite
import io
import os
import re
import plotly.graph_objects as go
from PIL import Image
from cairosvg import svg2png
import helpers

# Build App
external_stylesheets = [dbc.themes.LUX]
app = dash.Dash(__name__, title='Digit doodle interpreter', external_stylesheets=external_stylesheets)
server = app.server
figure_config = {
    "responsive": True,
    "displaylogo": False,
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
        dbc.Navbar(children=html.H1("Digit doodle interpreter", style={'color':'white'}),
            dark=True, color='dark', style={'margin-bottom':0}),
        dbc.Container([
                # Left block
                dbc.Jumbotron(
                    [
                        dcc.Graph(id="figure-container", figure=helpers.new_figure(), config=figure_config,
                            style={'display':'flex', 'height': "50vh", 'margin-right':0}),
                        dbc.Button("Clear canvas", id="clear-graph",
                            style={'display':'flex', 'float':'right', 'margin-top':10, 'margin-right':10}),
                        dbc.Button("What number is this?", id="save-image",
                            style={'display':'flex', 'float':'right', 'margin-top':10, 'margin-right':10}),
                    ],
                    style={'width':'60vw', 'height':'80vh', 'margin':10,
                        'margin-right':5, 'margin-left':0, 'background-color':'dimgrey'},
                    ),
                # Right block
                dbc.Jumbotron(
                    [
                        html.H3("Prediction", style={'display':'flex'}),
                        dbc.Container(
                            dcc.Markdown("Please write a number on the canvas"),
                            id='prediction-container',
                            style={'display':'inline-flex', 'height':'10vh'},
                            ),
                        html.H3("Prediciton overview", style={'margin-bottom':15, 'color':'gray'}),
                        dbc.Container(
                            dcc.Markdown(helpers.markdown_table(), style={"white-space": "pre"}),
                            id='table-container',
                            ),
                    ],
                    style={'width':'40vw', 'height':'80vh', 'padding-top':40,
                        'margin':10, 'margin-right':0, 'margin-left':5,},
                    ),
                ],
                style={'display':'inline-flex'},
                fluid=True,
                ),
    ],
    style={'content-align':'center'}
    )

# Load model
path = os.path.dirname(__file__)
model_path = os.path.join(path, 'model.tflite')
interpreter = tflite.Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# "What number is this?" button callback
@app.callback(
    Output("prediction-container", "children"),
    Output("table-container", "children"),
    Input("save-image", "n_clicks"),
    Input("clear-graph", "n_clicks"),
    State("figure-container", "relayoutData"),
)
def save_image(predict_trigger, clear_canvas_trigger, relayout_data):
    ctx = dash.callback_context
    for prop in ctx.triggered:
        if prop['prop_id'] == 'clear-graph.n_clicks' or not relayout_data:
            return dcc.Markdown("Please write a number on the canvas"), dcc.Markdown(helpers.markdown_table(), style={"white-space": "pre"})

    # else: Get data
    shapes=relayout_data["shapes"]
    paths=[]
    for shape in shapes:
        paths.append(shape["path"])
    img=helpers.path2img(paths, size=(28,28), stroke_width=4, viewbox_shift=(10,10), viewbox_margin=(20,20))
    input_data = np.array([helpers.img2array(img)], dtype=np.float32)

    # Run model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Fromat results
    prediction = np.argmax(output_data)
    confidence = '{:.2f}% confidence'.format(np.max(output_data)*100)
    string = helpers.markdown_table(output_data, prediction)

    return [html.H1(prediction, style={'margin-right':20, 'height':'50px', 'fontSize':'30'}), \
            html.H5(confidence, style={'color':'gray'})], \
            dcc.Markdown(string, style={"white-space": "pre"})

# "Clear canvas" button callback
@app.callback(
    Output("figure-container", "figure"),
    Output("figure-container", "relayoutData"),
    Input("clear-graph", "n_clicks"),
)
def clear_figure(n_clicks):
    return helpers.new_figure(), {}


if __name__ == '__main__':
    app.run_server(debug=True)
