import tensorflow as tf
import tensorflow_text as text
import tensorflow_addons as tfa
from sklearn.metrics import f1_score
from numpy import argmax, array
import matplotlib.pyplot as plt

from load_data import split_binary_data


DATASET_TRAIN_PATH      = "PATH_TO_THE_DATA"
SAVED_MODEl_PATH        = "PATH_TO_THE_SAVED_MODEL"
LEARNING_RATE           = 2e-5 
SMOOTHING               = 0.0


f1_macro = tfa.metrics.F1Score(num_classes = 1, threshold = 0.5, average = "macro", name = 'f1_macro')
f1_weighted = tfa.metrics.F1Score(num_classes = 1, threshold = 0.5, average = "weighted", name = 'f1_weighted')
f1_micro = tfa.metrics.F1Score(num_classes = 1, threshold = 0.5, average = "micro", name = 'f1_micro')

reloaded_model = tf.keras.models.load_model(SAVED_MODEl_PATH, custom_objects = {'f1_macro': f1_macro, 'f1_weighted': f1_weighted, 'f1_micro': f1_micro})

loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = SMOOTHING)
optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE)

reloaded_model.compile(optimizer = optimizer, loss = loss, metrics = [f1_macro, f1_weighted, f1_micro, tf.metrics.CategoricalAccuracy()])

X_train, X_validation, X_test, y_train, y_validation, y_test = split_binary_data(DATASET_TRAIN_PATH)

# test_result = reloaded_model.evaluate(x = X_test, y = y_test, return_dict = True)
# print(f'Test Result: {test_result}')

y_pred = reloaded_model.predict(X_test)

y_pred = [round(y[0]) for y in y_pred]

print(f"Macro F1: {f1_score(y_test, y_pred, average = 'macro')*100:.2f}%")
print(f"Weighted F1: {f1_score(y_test, y_pred, average = 'weighted')*100:.2f}%")
print(f"Micro F1: {f1_score(y_test, y_pred, average = 'micro')*100:.2f}%")
