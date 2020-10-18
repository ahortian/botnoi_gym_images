# pip install --upgrade pip
# pip install --upgrade tensorflow
# pip install tensorflow_hub
# pip install keras
# pip install Pillow

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import urllib

import keras
from keras.preprocessing import image


# Create pipeline
def predictImage(pic_name):
    # Labels
    dataset_labels = [
        'Elliptical',
        'Lateral Pull Down',
        'Roman Chair Bench',
        'Squat Rack',
        'Stationary Bicycle',
        'Treadmill'
        ]

    #Load model
    modFile = 'mygym_model.h5'
    mod = tf.keras.models.load_model(modFile,custom_objects={'KerasLayer':hub.KerasLayer})
    # input image
    img = image.load_img(pic_name, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.    

    y_prob = mod.predict(img_tensor)
    y_classes = y_prob.argmax(axis=-1)
    res = dataset_labels[y_classes[0]]
    return res, y_prob


def loadImage(URL):
    img_path = 'temp.jpg'
    with urllib.request.urlopen(URL) as url:
        with open(img_path, 'wb') as f:
            f.write(url.read())
    return img_path
  
def predictImageFromURL(img_url):
    img_path = loadImage(img_url)
    res, y_prob = predictImage(img_path)
    return res, y_prob

##--------------------------------##
##----------- Test ---------------##
##--------------------------------##
#print('The result is: {}.'.format(predictImage('uploads/image1.jpg')))

# img_url = 'https://lifefitness.co.uk/product-photos/7521/frame_color:7016,upholstery_color:516/'
# print('The result is: {}.'.format(predictImageFromURL(img_url)))
