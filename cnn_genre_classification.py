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

    #tensor flow aspect 3d array for each sample, now each is 2d
    #3d array -> (130,13,1) in this case the third channel is just 1 (like if we have grey image and not RGB images which would have 3 channels)

    X_train = X_train[...,np.newaxis] # 4d array -> (num_samples,130,13,1)    (we obtained a 4d array, so with 4 dimensions)
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    #returns all 4d arraies
    return X_train, X_validation, X_test, y_train, y_validation, y_test



def build_model(input_shape):
    #create the model
    model = keras.models.Sequential()

    #1st conv layer
    model.add(keras.layers.Conv2D(32,
                                  kernel_size=(3,3),
                                  activation= "relu",
                                  input_shape=input_shape))
    model.add(keras.layers.MaxPool2D(pool_size=(3,3),
                                     strides=(2,2),
                                     padding="same"))
    model.add(keras.layers.BatchNormalization()) #this layer speed up the training (the convergence will be faster) and the model will be more reliable

    #2nd conv layer
    model.add(keras.layers.Conv2D(32,
                                  kernel_size=(3, 3),
                                  activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3),
                                     strides=(2, 2),
                                     padding="same"))
    model.add(keras.layers.BatchNormalization())

    #3d conv layer
    model.add(keras.layers.Conv2D(32,
                                  kernel_size=(2, 2),
                                  activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2),
                                     strides=(2, 2),
                                     padding="same"))
    model.add(keras.layers.BatchNormalization())

    #flatten the output of the conv and feed it into the dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3)) #this Dropout layer is inserted to avoid overfitting

    #output layer
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


def predict(mode,X,y):
    # X is a 3d array X-> (130,13,1), but predict expect a 4 dimensions array (1,130,13,1)
    X = X[np.newaxis,...]

    #prediction = [ [0.1 , 0.2 , ...]]
    #the real prediction is obtained by extracting it from the prediction
    prediction = model.predict(X)
    #extract index with max value
    prediction_index = np.argmax(prediction, axis =1)   #one dimensional array with the index of the predicted value (disco, raggae..)
    print("Expected index: {}, Predicted index: {}".format(y,prediction_index))




#create train, validation and test sets (cross validation)
X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25,0.2)    #(X : input, y: target)

#build the CNN network (with a custom function that incapsulate and modularize the implementation
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
model = build_model(input_shape)

#compile the network
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

#train the CNN
history = model.fit(X_train, y_train, validation_data=(X_validation,y_validation), batch_size= 32 , epochs= 60)

#evalutate the CNN with the test set
test_error, test_accuracy = model.evaluate(X_test, y_test, verbose =1)
print("accuracy on test set is: {}".format(test_accuracy))

#make some predictionson a sample (we pick some random values from test data)
X = X_test[100]
y= y_test[100]
predict(model,X,y)