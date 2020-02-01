from flask import Flask, redirect, url_for, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import tensorflow 
# Import Keras dependencies
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image


# Import other dependecies
import numpy as np
import h5py
from PIL import Image
import PIL
import os

import pandas as pd

with open('./model/model_config.json') as json_file:
    json_config = json_file.read()
    
model = tensorflow.keras.models.model_from_json(json_config)
model.load_weights('./model/model_weights.h5')

data_table = pd.read_csv('./artist_table.csv')


classes = list(data_table['name']) 
bio = list(data_table['bio'])


def model_predict(image_path, model):

    image = tensorflow.io.read_file(image_path)
    
    image = tensorflow.image.decode_png(image, channels= 3)

    image = tensorflow.cast(image, tensorflow.float32)
  
    image = (image / 255.0)

    image = tensorflow.image.resize(image, size=(299,299))
  
    predictions = model.predict(image[tensorflow.newaxis,...])

    predictions =   [np.argmax(predictions)]
    
    return predictions 




path = './static/images'


app = Flask(__name__)


@app.route('/', methods =['GET'])

def index():
    return render_template('index.html')        
     

@app.route('/', methods =['GET' , 'POST']) 
def upload():


    if request.method == 'POST':

        f = request.files['file']

        # Save the file to ./uploads 
        file_path = os.path.join(path, secure_filename(f.filename))

        f.save(file_path)

        prediction = model_predict(file_path, model)

        predicted_class = classes[prediction[0]]

        predicted_bio = bio[prediction[0]]       

        file_path_origi = os.path.join('./images', secure_filename(f.filename))

        return render_template('predict.html', file=(predicted_class, file_path_origi, predicted_bio))        



if __name__ == '__main__':
    app.run(debug = True)

classes = ['Leonardo_da_Vinci',
 'Albrecht_Dürer',
 'Rembrandt',
 'Mikhail_Vrubel',
 'Vincent_van_Gogh',
 'Henri_Matisse',
 'Pablo_Picasso',
 'Peter_Paul_Rubens',
 'Amedeo_Modigliani',
 'Pierre-Auguste_Renoir',
 'Marc_Chagall',
 'Sandro_Botticelli',
 'Paul_Klee',
 'Andy_Warhol',
 'Paul_Gauguin',
 'Alfred_Sisley',
 'Titian',
 'Rene_Magritte',
 'Francisco_Goya',
 'Edgar_Degas']
