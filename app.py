from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow import expand_dims
import numpy as np
import os
import mahotas as mh


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")

class_dict = {'Normal': 0, 'Tuberculosis': 1}

def predict_label(img_path):
    image = mh.imread(img_path)   
    IMM_SIZE = 224

    if len(image.shape) > 2:
        image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE, image.shape[2]]) 
    else:
        image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE])
    if len(image.shape) > 2:
        image = mh.colors.rgb2grey(image[:,:,:3], dtype = np.uint8)

    image = np.array(image)/255
    
    image = image.reshape(-1, IMM_SIZE, IMM_SIZE, 1)
    
    preds = model.predict(image)
    preds=np.argmax(preds,axis=1)
    preds = preds.reshape(1,-1)[0]
    
    diag = {i for i in class_dict if class_dict[i]==preds}
    
    return diag

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            prediction = predict_label(img_path)
            return render_template('index.html', uploaded_image=image.filename, prediction=prediction)

    return render_template('index.html')

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)