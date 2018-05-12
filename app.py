from flask import Flask, redirect, request, jsonify,render_template
from keras import models
import numpy as np
from PIL import Image
import io
from keras.models import model_from_json
import tensorflow as tf

app = Flask(__name__)
model = None
graph = tf.get_default_graph()
#global model, graph
#model, graph = init()

def load_model():
    global model
    keras_model="Save_model.json"
    keras_param="Save_model.hdf5"
    model = model_from_json(open(keras_model).read())
    model.load_weights(keras_param)
    model.summary()
    #print('Loaded the model')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.files and 'picfile' in request.files:

        img = request.files['picfile'].read()
        img = Image.open(io.BytesIO(img))
        img.save('test.jpg')
        img=img.resize((50,50))
        img = np.asarray(img) / 255.
        img = np.expand_dims(img, axis=0)
        global graph
        with graph .as_default():

            pred = model.predict(img)
            persons = [
                'ちょまど',
                '池澤あやか',
                '石原さとみ',
                '剛力彩芽'
            ]

            confidence = int(round(max(pred[0]), 3)*100)
            pred = persons[np.argmax(pred)]

        data = dict(pred=pred, confidence=str(confidence))
        return jsonify(data)

    return 'Picture info did not get saved.'


@app.route('/currentimage', methods=['GET'])
def current_image():
    fileob = open('test.jpg', 'rb')
    data = fileob.read()
    return data


if __name__ == '__main__':
    load_model()
    app.run()
