import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow.keras  as keras

DATASET_PATH = "data.json"

def load_data(dataset_path):
    with open(dataset_path,"r") as fp:
        data = json.load(fp)

    #convert lists in numpy array
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs,targets

def prepare_datasets(test_size, validation_size):
    #load data
    x,y = load_data(DATASET_PATH)        #take the already defined function in the previus module
    #create the train/test split
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=test_size)

    #create the train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train)

    #returns all 4d arraies
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    #generate RNN-LSTM model
    model = keras.Sequential()

    #2 LSTM layers
    model.add(keras.layers.LSTM(64,input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    #dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))      #layer to avoid or mitigate overfitting (0.3 = 30% of dropout during training)

    #output layer
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model

#def plot_history():


#create train, validation and test sets (cross validation)
X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25,0.2)    #(X : input, y: target)

#create the network
input_shape = (X_train.shape[1], X_train.shape[2])       #input shape bidimensional, we drop the third dimension which is not useful (130,13)
model = build_model(input_shape)

#compile the network
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

#train the CNN
history = model.fit(X_train, y_train, validation_data=(X_validation,y_validation), batch_size= 32 , epochs= 60)

#plot accuracy/error for training and valiadtion
#plot_history(history)

#evaluate model and test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy: ',test_acc)