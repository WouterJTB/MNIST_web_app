import io
import re
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from cairosvg import svg2png

def new_figure(size=(400,400), tickrange=(100,100), viewbox_shift=(-1,-1), viewbox_margin=(2,2)):
    fig = go.Figure(layout={"dragmode":"drawopenpath", "height":size[0], "width":size[1]})
    fig.update_xaxes(range=[0,tickrange[0]], showticklabels=False, showgrid=False)
    fig.update_yaxes(range=[0,tickrange[1]], showticklabels=False, showgrid=False)
    return fig

def path2img(shape, size=(100,100), stroke_width=0.5, viewbox_shift=(-1,-1), viewbox_margin=(2,2)):
    """
    Return a PIL.Image object for a dcc.Graph relayoutData shape.

    Input:
    dict shape
    tuple size - target size in pixels default: (100,100)
    float stroke_width - relative stroke width for the svg path, default: 0.5

    Output:
    PIL.Image object - 1 channel, pixel intensity of 0 or 255
    """
    d=shape["path"]
    paths=np.array([s.split(',') for s in list(re.findall(r'(\d+\.?\d*\,\d+\.?\d*)',d))],dtype=np.float32)
    minpath_x,minpath_y=min(paths[:,0]),min(paths[:,1])
    viewbox_width=str(max(paths[:,0]) - minpath_x + viewbox_margin[0])
    viewbox_height=str(max(paths[:,1]) - minpath_y + viewbox_margin[1])
    width=str(size[0])
    height=str(size[1])
    path_id="svg_1"
    color=shape["line"]["color"]
    svg=[
        '<svg width="'+width+'" height="'+height+'" \
        xmlns="http://www.w3.org/2000/svg" \
        viewBox="'+str(minpath_x - viewbox_shift[0])+' '+str(minpath_y - viewbox_shift[1])+' \
        '+viewbox_width+' '+viewbox_height+'">',
        '<path d="'''+d+'''" id="'''+path_id+'''" \
        stroke-width="'''+str(stroke_width)+'''" stroke="'''+color+'" fill="none"/>',
        '</svg>']
    in_mem_file = io.BytesIO()
    svg2png(''.join(svg).encode("UTF-8"), write_to=in_mem_file)
    img=Image.open(io.BytesIO(in_mem_file.getvalue())).transpose(Image.FLIP_TOP_BOTTOM)
    img=img.convert('L').point(lambda p: 0 if p==0 else 255)
    return img

def img2array(img, resize=(28,28)):
    arr=np.array(img.resize((resize[0],resize[1]), Image.LANCZOS))
    arr=arr/255
    arr=arr[:,:,np.newaxis]
    return arr
