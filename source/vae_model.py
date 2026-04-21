
#a supprimer
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, models, datasets, callbacks
from sklearn.model_selection import train_test_split

#-------------------------------------------------------------------------------------------------------------------
#import the data

path = 'img_align_celeba/'
batch_size = 64 

def process_path(file_path):
    # Loads, resizes to 128x128, and scales to 0-1
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [128, 128])
    img = img / 255.0 #We normalize it to have values between 0 and 1
    return img, img  # Returns (x, y) where x=y for your autoencoder

def import_data(path, batch_size):
    #Get file paths
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]
    
    #Split the paths (not the images)
    train_files, test_files = train_test_split(files, test_size=0.3, random_state=30)
    
    #Create datasets that load and process images only when we need them
    train_ds = tf.data.Dataset.from_tensor_slices(train_files).map(process_path).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices(test_files).map(process_path).batch(batch_size)
    
    return train_ds, test_ds

# These datasets can be passed directly to model.fit() without loading all images into memory at once:
# model.fit(x_train, validation_data=x_test, epochs=10)
x_train, x_test = import_data(path, batch_size)

#-------------------------------------------------------------------------------------------------------------------
#Parameters

img_size=128
channels=3 # 3 for colors
batch_size=1000 #size for the training
buffer_size=1000 #size of the buffer for the datamix
embedding_dim=2
epochs=3

#------------------------------------------------------------------------------------------------------------------
#Encoder

encoder_input=layers.Input(shape=(img_size, img_size, channels), name='encoder_input')

##Different layers
x=layers.Conv2D(128,(3,3), strides=2, activation='relu', padding='same')(encoder_input)
x=layers.Conv2D(64,(3,3), strides=2, activation='relu', padding='same')(x)
x=layers.Conv2D(32,(3,3), strides=2, activation='relu', padding='same')(x)
x=layers.Conv2D(16,(3,3), strides=2, activation='relu', padding='same')(x)
x=layers.Conv2D(8,(3,3), strides=2, activation='relu', padding='same')(x)
shape_before_flattening=K.int_shape(x)[1:]#Save the dimension for the decoder
x=layers.Flatten()(x)

encoder_output=layers.Dense(embedding_dim, name="encoder_output")(x) #Dense layer compressed representation of dim embedding

encoder=models.Model(encoder_input, encoder_output)#creation of the model

#------------------------------------------------------------------------------------------------------------------
#decoder

decoder_input=layers.Input(shape=(embedding_dim,),name="decoder_input") #definition of the input

x=layers.Dense(int(np.prod(shape_before_flattening)))(decoder_input)#dense layer to prepare output
x=layers.Reshape(shape_before_flattening)(x)#Same dimension as before
x=layers.Conv2DTranspose(8,(3,3), strides=2, activation="relu", padding="same")(x)
x=layers.Conv2DTranspose(16,(3,3), strides=2, activation="relu", padding="same")(x)
x=layers.Conv2DTranspose(32,(3,3), strides=2, activation="relu", padding="same")(x)
x=layers.Conv2DTranspose(16,(3,3), strides=2, activation="relu", padding="same")(x)
x=layers.Conv2DTranspose(8,(3,3), strides=2, activation="relu", padding="same")(x)

decoder_output=layers.Conv2D(channels,(3,3),strides=1, activation="sigmoid", padding="same", name="decoder_output",)(x) #Output layer

decoder=models.Model(decoder_input, decoder_output)
#-----------------------------------------------------------------------------------------------------------------
# Autoencoder
autoencodeur = models.Model(encoder_input, decoder(encoder_output), name='autoencoder')

# Compilation and Build
autoencodeur.compile(optimizer='adam', loss='binary_crossentropy')
autoencodeur.build(input_shape=(None, img_size, img_size, channels))

# Training the autoencoder
#autoencodeur.fit(x_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=x_test)

for x, _ in x_test.take(1):
    reconstructed = autoencodeur.predict(x)

    for i in range(5):
        plt.figure(figsize=(4,2))

        # Original
        plt.subplot(1,2,1)
        plt.imshow(x[i])
        plt.title("Original")
        plt.axis("off")

        # Reconstruite
        plt.subplot(1,2,2)
        plt.imshow(reconstructed[i])
        plt.title("Reconstruit")
        plt.axis("off")

        plt.show()


