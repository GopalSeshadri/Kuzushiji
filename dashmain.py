import numpy as np
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go
import dash
import dash_core_components as core
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import sys
import pickle
from keras.models import model_from_json
from keras import backend as K
import base64
from PIL import Image
from keras.preprocessing.image import array_to_img, img_to_array

app = dash.Dash()
server = app.server

DATA_NAME = 'k49'
image_filename = 'assets/1.png'

test_x = np.load('Data/{}/{}-test-imgs.npz'.format(DATA_NAME.upper(), DATA_NAME.lower()))['arr_0']
test_y = np.load('Data/{}/{}-test-labels.npz'.format(DATA_NAME.upper(), DATA_NAME.lower()))['arr_0']

mapping_df = pd.read_csv('Data/{}/{}_classmap.csv'.format(DATA_NAME.upper(), DATA_NAME.lower()))

app.layout = html.Div([
    html.Div([
        html.H1('Kuzushiji Hiragana Classifier', style = {'fontFamily' : 'Helvetica',
                                                        'textAlign' : 'center',
                                                        'width' : '100%'})
    ], style = {'display' : 'flex'}),

    html.Div([
        html.Div([
            html.H3('Calligraphically Written Hiragana',
                style = {'font' : {'size' : 16},
                                'display' : 'inline-block',
                                'float' : 'left'})
        ]),

        html.Div([
            html.Img(id = 'display-image',
                src = image_filename)
        ], style = {'paddingLeft' : '50px'}),

        html.Div([
            html.Button(id = 'submit-button', children = 'Classify',
                        style = {'width' : '80px',
                                'height' : '30px'})
        ], style = {'width' : '100%',
                    'paddingLeft' : '50px'})

    ], style = {'width' : '100%',
                'height' : '80%',
                'display' : 'flex',
                'float' : 'left',
                'paddingLeft' : '50px',
                'paddingTop' : '50px'}),

    html.Div([

        html.H3('Predicted Hiragana',
            style = {'fontSize' : 16, 'fontFamily' : 'Helvetica',
                    'display' : 'inline-block',
                    'float' : 'left',
                    'paddingRight' : '50px'}),

        core.Input(id = 'predicted-text',
                    placeholder = 'Predicted hiragana ...',
                    style = {'display' : 'inline-block',
                            'float' : 'left',
                            'paddingLeft' : '50px'}),

        html.H3('Actual Hiragana',
            style = {'fontSize' : 16, 'fontFamily' : 'Helvetica',
                    'display' : 'inline-block',
                    'float' : 'left',
                    'paddingLeft' : '50px',
                    'paddingRight' : '50px'}),

        core.Input(id = 'actual-text',
                    placeholder = 'Actual hiragana ...',
                    style = {'display' : 'inline-block',
                            'float' : 'left',
                            'paddingLeft' : '50px'})

    ], style = {'width' : '100%',
                'display' : 'flex',
                'float' : 'left',
                'paddingLeft' : '50px',
                'paddingTop' : '50px'})

], style = {'fontFamily' : 'Helvetica',
            'width' : '100%',
            'height' : '100%',
            'float' : 'left'})

def findPrediction(model_name, input_img):
    '''
    This function takes in the model name and the normalised array of input image and returns the prediction.
    Parameters:
    model_name (str) : The name of the model
    input_img (numpy array) : The normalised array of input image
    Returns:
    pred_class (int) : The class of the prediction, it will be an integer from 0 to 48.
    '''
    input_img = input_img.reshape(-1, 28, 28, 1)
    print(input_img.shape, file=sys.stderr)
    json_file = open('Models/{}.json'.format(model_name.lower()), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("Models/{}.h5".format(model_name.lower()))
    prediction = loaded_model.predict(input_img)
    pred_class = np.argmax(prediction)
    K.clear_session()
    return pred_class

@app.callback([Output(component_id = 'display-image', component_property = 'src'),
            Output(component_id = 'predicted-text', component_property = 'value'),
            Output(component_id = 'actual-text', component_property = 'value')],
            [Input(component_id = 'submit-button', component_property = 'n_clicks')])
def affectDisplayImage(n_clicks):
    idx = np.random.randint(0, len(test_y))
    actual = mapping_df[mapping_df['index'] == test_y[idx]]['char']
    image_filename = 'assets/{}.png'.format(idx)
    print(test_x[idx].shape, file=sys.stderr)
    pred_class = findPrediction('lenet', test_x[idx])
    prediction = mapping_df[mapping_df['index'] == pred_class]['char']
    return image_filename, actual, prediction



if __name__ == '__main__':
    app.run_server()
