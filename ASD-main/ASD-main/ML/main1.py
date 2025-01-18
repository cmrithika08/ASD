import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from skimage import transform
import pickle


photo_size=224


def prepare_dataset(data_dir):
    datagen = ImageDataGenerator(
        rescale= 1/255,
        rotation_range=40,
        width_shift_range=.2,
        height_shift_range=.2,
        shear_range=.1,
        horizontal_flip=True,
        fill_mode='nearest',
        zoom_range=.2,
    )
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(photo_size,photo_size),
        class_mode='binary',
        batch_size=128,
        classes=['non_autistic','autistic']
    )
    return generator


train_data=prepare_dataset("C:\\Users\\PRIYANKA A H\\Downloads\\Deployment\\pickle\\AutismDataset\\AutismDataset\\train")
validation_data = prepare_dataset("C:\\Users\\PRIYANKA A H\\Downloads\\Deployment\\pickle\\AutismDataset\\AutismDataset\\valid")
test_data=prepare_dataset("C:\\Users\\PRIYANKA A H\\Downloads\\Deployment\\pickle\\AutismDataset\\AutismDataset\\test")


validation_data.class_indices




#get_ipython().system('pip install efficientnet')

def use_efficient_net(model_type='B0'):
    from tensorflow.keras.optimizers import RMSprop
    from keras.models import Model
    import efficientnet.tfkeras as efn
    if model_type=='B0':
        efn_model = efn.EfficientNetB0(input_shape = (photo_size, photo_size, 3), include_top = False, weights = 'imagenet')
    else:
        efn_model = efn.EfficientNetB7(input_shape = (photo_size, photo_size, 3), include_top = False, weights = 'imagenet')
    for layer in efn_model.layers:
        layer.trainable = False
    #
    x = efn_model.output
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    # Add a final sigmoid layer with 1 node for classification output
    predictions = Dense(1, activation="sigmoid")(x)
    efficient_net = Model(efn_model.input,predictions)

    efficient_net.compile(RMSprop(learning_rate=0.0001, decay=1e-6),loss='binary_crossentropy',metrics=['accuracy'])
    return efficient_net
efficient_net=use_efficient_net('B0')
efficient_net.summary()


effb0_history = efficient_net.fit(train_data, validation_data = validation_data, epochs = 50)
efficient_net.save("efficient_net_B0_model.h5")

pickle.dump(efficient_net,open("image.pkl",'wb'))


efficient_net.evaluate(test_data)








