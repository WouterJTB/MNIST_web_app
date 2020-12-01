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

def path2img(path_list, size=(100,100), stroke_width=0.5, viewbox_shift=(-1,-1), viewbox_margin=(2,2)):
    """
    Function used to convert svg paths into a PIL.Image object.

    Input:
    path_list (list)            list of 'd' values of SVG paths, i.e. list of strings like "M1.00,2.00L5.00,6.00..."
    size (tuple)                target output size in pixels default: (100,100)
    stroke_width (int/float)    relative stroke width for the svg path, default: 0.5
    viewbox_shift (tuple)       tuple indicating the x,y shift of the viewbox, can be used to recenter the image
    viewbox_margin (tuple)      tuple indicating the width,height margin of the viewbox compared to the shapes,
                                determines how the image is cropped

    Output:
    PIL.Image object            'L' channel image, with reversed pixel intensity
    """
    pathstring = ''.join(path_list)
    paths=np.array([s.split(',') for s in list(re.findall(r'(\d+\.?\d*\,\d+\.?\d*)',pathstring))],dtype=np.float32)
    minpath_x,minpath_y=min(paths[:,0]),min(paths[:,1])
    viewbox_width=str(max(paths[:,0]) - minpath_x + viewbox_margin[0])
    viewbox_height=str(max(paths[:,1]) - minpath_y + viewbox_margin[1])
    width=str(size[0])
    height=str(size[1])
    svg=[('<svg xmlns="http://www.w3.org/2000/svg" '
        'width="'+width+'" height="'+height+'" '
        'viewBox="'+str(minpath_x - viewbox_shift[0])+' '+str(minpath_y - viewbox_shift[1])+' '
        ''+viewbox_width+' '+viewbox_height+'">')]
    for p in path_list:
        svg.append('<path d="'+p+'" stroke-width="'+str(stroke_width)+'" stroke="black" fill="none"/>')
    svg.append('</svg>')
    in_mem_file = io.BytesIO()
    svg2png(''.join(svg).encode("UTF-8"), write_to=in_mem_file)
    png=Image.open(io.BytesIO(in_mem_file.getvalue())).transpose(Image.FLIP_TOP_BOTTOM)
    img=Image.new("RGB", png.size, "WHITE")
    img.paste(png, (0,0), png)
    img=img.convert('L').point(lambda px: 255-px)
    return img

def img2array(img, resize=(28,28)):
    """
    Function used to convert a single channel PIL.Image object to a numpy array.
    Array has normalized pixel values for model inference/training.

    Input:
    img (PIL.Image object)      Single channel PIL.Image object to convert to numpy array
    resize (tuple)              tuple to indicate the target size of the image in pixels

    Output:
    numpy array (ndarray)       Array containing the normalized pixel intensity values
    """
    arr=np.array(img.resize((resize[0],resize[1]), Image.LANCZOS))
    arr=arr/255
    arr=arr[:,:,np.newaxis]
    return arr
