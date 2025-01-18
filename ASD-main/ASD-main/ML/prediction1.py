from tensorflow.keras.applications import EfficientNetB0
from keras.models import load_model

import keras

# Define custom objects
class FixedDropout(keras.layers.Dropout):
    def get_config(self):
        config = super().get_config()
        return config

# Register the custom layer
keras.utils.get_custom_objects()['FixedDropout'] = FixedDropout

inception_model= load_model('inception_model.h5')

from PIL import Image
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

photo_size=224
def load_image_from_path(filename):
    img = mpimg.imread(filename)
    imgplot = plt.imshow(img)
    plt.show()
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32') / 255
    np_image = transform.resize(np_image, (photo_size, photo_size, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

plt.plot([1, 2, 3, 4])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Sample Plot')
plt.show()

'''import os
# mTestPath = "/AutismDataset/AutismDataset/train/Autistic/Autistic.990.jpg"
# for test in os.listdir(mTestPath):
    
    
import glob

mTestPath = "C:\\Users\\PRIYANKA A H\\Downloads\\Deployment\\pickle\\AutismDataset\\AutismDataset\\test\\Autistic\\Autistic.90.jpg"
for test in glob.glob(mTestPath + '/*.jpg'):
    #print(test)    
    print(test)
    img = load_image_from_path(os.path.join(mTestPath, test))

    res = efficientnet_b0_model.predict(img).argmax()
    if(res==1):
        print("Non-Autistic")
    else:
        print("Autistic")
    '''
# Load and preprocess the unseen image
image_path = "C:\\Users\\PRIYANKA A H\\Downloads\\Deployment\\pickle\\AutismDataset\\AutismDataset\\test\\Autistic\\Autistic.99.jpg"
input_image = load_image_from_path(image_path)

# Make predictions on the preprocessed image
res = inception_model.predict(input_image).argmax()

# Interpret the prediction results
if res==1:
    print("Predicted class: Autistic")
else:
    print("Predicted class: Non-Autistic")
