# MNIST Web App
Project is currenlty in progress.

Interactive web application that allows users to write a digit which is then classified by the model. 
The front end was made in plotly Dash, for the classification, a convolutional neural network was trained on the MNIST dataset using TensorFlow.

To run the app locally, download [app.py](app.py) and [back_end.py](back_end.py) and install the dependencies listed in 'Web App Dependencies'. Then run app with: 

    $ python app.py


## Main files
* [app.py](app.py) - The main application
* [back_end.py](back_end.py) - File with helper functions
* [model_training_notebook.ipynb](model_training_notebook.ipynb) - Colab notebook in which the model was trained
* [model.tflite](model.tflite) - Trained model

## Datasets
[The MNIST Database of handwritten digits](http://yann.lecun.com/exdb/mnist/)

## Web App Dependencies
* Python 3.8
* CairoSVG 2.5.0
* dash 1.17.0
* dash-bootstrap-components 0.10.7
* numpy 1.19.3
* Pillow 8.0.1
* plotly 4.13.0
* tflite-runtime [2.5.0-cp38-cp38-win_amd64](https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp38-cp38-win_amd64.whl)*

Full list of libraries is included in [requirements.txt](requirements.txt)

*The app was made in Windows 10 64-bit. Linux or OSx environments should install a [different version of the tflite interpreter](https://www.tensorflow.org/lite/guide/python)

## To Do's
* Make front end pretty
* Improve model performance
