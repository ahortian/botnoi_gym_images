# pip install --upgrade pip
# pip install --upgrade tensorflow
# pip install tensorflow_hub
# pip install keras
# pip install Pillow


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

import keras
from keras.preprocessing import image

# def main():
#     # Load Model
#     modFile = 'mygym_model.h5'

#     mod = tf.keras.models.load_model(modFile,custom_objects={'KerasLayer':hub.KerasLayer})
#     #print(mod.summary())

#     dataset_labels = np.array(['Gym Machine Lateral Pull Down Cable',
#        'Gym Machine Roman Chair Bench', 'Gym Machine Squat Rack',
#        'Gym Machine Stationary Bicycle'])
#     #print(dataset_labels)

#     #img_path = 'uploads/image1.jpg'
#     img_path = 'uploads/image2.jpg'
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_tensor = image.img_to_array(img)
#     img_tensor = np.expand_dims(img_tensor, axis=0)
#     img_tensor /= 255.

#     y_prob = mod.predict(img_tensor)
#     y_classes = y_prob.argmax(axis=-1)

#     #print (y_prob)
#     #print (y_classes, dataset_labels[y_classes])
#     print (dataset_labels[y_classes])


# if __name__ == '__main__':
#     main()


# Create pipeline
def predictImage(pic_name):
    # Labels
    dataset_labels = ['Lateral Pull Down',
       'Roman Chair Bench', 
       'Squat Rack',
       'Stationary Bicycle']

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
    return res


#print('The result is: {}.'.format(predictImage('uploads/image1.jpg')))