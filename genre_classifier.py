import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATASET_PATH = "data.json"

def load_data(dataset_path):
    with open(dataset_path,"r") as fp:
        data = json.load(fp)

    #convert lists in numpy array
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs,targets


#load data
inputs, targets = load_data(DATASET_PATH)

#split data into train and test set
inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs,
                                                                          targets,
                                                                          test_size=0.3)

#build the network architecture
model = keras.models.Sequential()

#input layer
model.add(keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])))
#hidden layers
model.add(keras.layers.Dense(512, activation="relu"))
model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dense(64, activation="relu"))
#output layer
model.add(keras.layers.Dense(10, activation="softmax")) #softmax is the activation function for the output layer in classification


#compile the network
optimizer = keras.optimizers.Adam(learning_rate = 0.0001)
model.compile(optimizer = optimizer,
              loss= "sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()


#train the network
model.fit(x = inputs_train,
          y = targets_train,
          validation_data=(inputs_test,targets_test),
          epochs = 50,
          batch_size= 32)





