import tensorflow as tf
import tensorflow_text as text
import tensorflow_addons as tfa
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from numpy import argmax, array
import matplotlib.pyplot as plt

from load_data import split_data


DATASET_TRAIN_PATH      = "PATH_TO_THE_DATA"
SAVED_MODEl_PATH        = "PATH_TO_THE_SAVED_MODEL"
LEARNING_RATE           = 2e-5 
SMOOTHING               = 0.0


f1_macro = tfa.metrics.F1Score(num_classes = 12, average = "macro", name = 'f1_macro')
f1_weighted = tfa.metrics.F1Score(num_classes = 12, average = "weighted", name = 'f1_weighted')
f1_micro = tfa.metrics.F1Score(num_classes = 12, average = "micro", name = 'f1_micro')

reloaded_model = tf.keras.models.load_model(SAVED_MODEl_PATH, custom_objects = {'f1_macro': f1_macro, 'f1_weighted': f1_weighted, 'f1_micro': f1_micro})

loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = SMOOTHING)
optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE)

reloaded_model.compile(optimizer = optimizer, loss = loss, metrics = [f1_macro, f1_weighted, f1_micro, tf.metrics.CategoricalAccuracy()])

X_train, X_validation, X_test, y_train, y_validation, y_test, labels = split_data(DATASET_TRAIN_PATH)

# test_result = reloaded_model.evaluate(x = X_test, y = y_test, return_dict = True)
# print(f'Test Result: {test_result}')

y_pred = reloaded_model.predict(X_test)

y_pred = array([labels[y] for y in argmax(y_pred, axis = 1)])
y_test = array([labels[y] for y in argmax(y_test, axis = 1)])

conf_matrix = confusion_matrix(y_test, y_pred, labels = labels)
# print(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = labels)
disp.plot()

plt.xticks(rotation = 45)

plt.show()
