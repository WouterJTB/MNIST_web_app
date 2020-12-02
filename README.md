# MNIST Web App
Project is currenlty in progress.

Interactive web application that allows users to write a digit which is then classified by the model.
The front end was made in plotly Dash, for the classification, a convolutional neural network was trained on the MNIST dataset using TensorFlow.

To run the app locally, download the files in the app folder and install the dependencies listed in 'Web App Dependencies'. Then run app with:

    $ python app.py


## Main files
* [app.py](app/app.py) - The main application
* [helpers.py](app/helpers.py) - File with helper functions
* [model_training_notebook.ipynb](model_training_notebook/model_training_notebook.ipynb) - Colab notebook in which the model was trained
* [model.tflite](app/model.tflite) - Trained model

## Datasets
[The MNIST Database of handwritten digits](http://yann.lecun.com/exdb/mnist/)

## Web App Dependencies
* Python 3.6-3.8
* CairoSVG
* dash
* dash-bootstrap-components
* numpy
* Pillow
* plotly
* tflite-runtime*

Full list of libraries is included in [requirements.txt](requirements.txt)

*tflite-runtime is specific to python release and OS, installation instructions can be found [here](https://www.tensorflow.org/lite/guide/python)

## To Do's
* Improve model performance
