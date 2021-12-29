import pandas as pd
import numpy as np
import cv2

from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import keras
from keras.models import Sequential
from keras.layers import Conv2D, Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping

# READ DATAS
limit = 16000

training_labels = pd.read_csv('data/train.csv', index_col=None)[:(limit)]
to_predict_ids = pd.read_csv('data/test.csv', index_col=None)[:(limit)]

train_images = [cv2.imread(file) for file in glob('data/train/*.png')[:limit]]
test_images = [cv2.imread(file) for file in glob('data/test/*.png')[:limit]]

# Convert to numpy array
train_images = np.array(train_images)
test_images = np.array(test_images)

# %%
# PREPROCESS DATAS
input_shape = train_images[0].shape
k_folds = 5

batchSize = 32
# Initialize data generator
train_dataGen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

train_size = int(len(training_labels) * 0.8)
test_size = len(training_labels) - train_size

training_labels['label'] = training_labels['label'].astype(str)

# Preprocess kaggle predictions photos
topredict_generator = train_dataGen.flow_from_dataframe(
    dataframe=to_predict_ids,
    directory="data/test",
    x_col="id",
    class_mode=None,
    shuffle=False,
    target_size=(128, 55),
    batch_size=batchSize)

# %%
from sklearn.metrics import accuracy_score, confusion_matrix

# Configure early stopping callbacks
esMax = EarlyStopping(monitor='accuracy', mode='max', min_delta=0.005, patience=7)
esMin = EarlyStopping(monitor='accuracy', mode='min', min_delta=0.005, patience=7)

model = Sequential()

#First Convolutional layer
model.add(Dense(units = 32, activation = 'relu', input_shape = input_shape))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(BatchNormalization())
#Flattening
model.add(Flatten())
#Hidden Layer
model.add(Dropout(0.2))
model.add(Dense(units = 64, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units = 128, activation = 'relu'))
model.add(BatchNormalization())
#Output Layer
model.add(Dense(units = 5 , activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fit the model and validate using K-Fold validation. K_FOLDS is initialized above with 5
for index in range(k_folds):
    print("================== Fold " + str(index + 1) + " =================== ")

    X_train, X_test, y_train, y_test = train_test_split(training_labels.iloc[:, :-1], training_labels.iloc[:, -1],
                                                        train_size=train_size, shuffle=True)
    # Concatenate ids and labels in a single dataframe to be processed with flow_from_dataframe
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    # Preprocess datas by Keras Data Generator
    train_generator = train_dataGen.flow_from_dataframe(
        dataframe=train,
        directory="data/train", x_col="id",
        y_col="label",
        class_mode="categorical",
        target_size=(128, 55),
        batch_size=batchSize)

    test_generator = train_dataGen.flow_from_dataframe(
        dataframe=test,
        directory="data/train", x_col="id",
        y_col="label",
        class_mode="categorical",
        target_size=(128, 55),
        batch_size=batchSize)
    model.fit(train_generator, epochs=50, steps_per_epoch=len(train) // batchSize)

    # Save model
    model.save_weights('model_fold' + str(index + 1) + '.h5')

    print("==== TEST DATA VALIDATION AND CONFUSION MATRIX FOR FOLD " + str(index + 1) + " ====")
    # Print validation metrics
    test_generator.shuffle = False

    y_true = np.array(test_generator.classes) + 1
    predictions = model.predict_generator(test_generator)
    y_pred = np.array([np.argmax(x) for x in predictions]) + 1

    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))

#### CREATING THE KAGGLE SUBMISSION FILE #####
test_images = np.array(test_images)

sample_preds = []

### Predict on test data ###

predictions = model.predict(topredict_generator)
sample_predictions = np.array([np.argmax(x) for x in predictions]) + 1

to_predict_ids = to_predict_ids.values.tolist()


def create_kaggle_file(file_name):
    print('----- CREATING KAGGLE SUBMISSION FORMAT ----')
    results = pd.DataFrame(columns=['id', 'label'])

    for i in range(len(to_predict_ids)):
        current_pred = {'id': to_predict_ids[i][0], 'label': sample_predictions[i]}
        results = results.append(current_pred, ignore_index=True)

    results = results.astype({'label': 'int32'})
    results.to_csv(file_name + '.csv', encoding='utf-8', index=False)


create_kaggle_file('BestSolution')
