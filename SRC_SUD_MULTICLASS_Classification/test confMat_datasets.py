import tensorflow as tf
import tensorflow_text as text
import tensorflow_addons as tfa
from numpy import array, argmax
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from load_data import split_data_test_dataset


DATASET_TRAIN_PATH      = "PATH_TO_THE_DATA"
SAVE_FIGURES_PATH       = "PATH_TO_THE_SAVED_FIGURES"
SAVED_MODEl_PATH        = "PATH_TO_THE_SAVED_MODELS"
LEARNING_RATE           = 2e-5 
SMOOTHING               = 0.0


f1_macro = tfa.metrics.F1Score(num_classes = 12, average = "macro", name = 'f1_macro')
f1_weighted = tfa.metrics.F1Score(num_classes = 12, average = "weighted", name = 'f1_weighted')
f1_micro = tfa.metrics.F1Score(num_classes = 12, average = "micro", name = 'f1_micro')

reloaded_model = tf.keras.models.load_model(SAVED_MODEl_PATH, custom_objects = {'f1_macro': f1_macro, 'f1_weighted': f1_weighted, 'f1_micro': f1_micro})

loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = SMOOTHING)
optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE)

reloaded_model.compile(optimizer = optimizer, loss = loss, metrics = [f1_macro, f1_weighted, f1_micro])

dictionary, labels = split_data_test_dataset(DATASET_TRAIN_PATH)

for dataset in dictionary.keys():

    X = array(dictionary[dataset][0])
    y_true = array(dictionary[dataset][1])

    y_pred = reloaded_model.predict(X)

    y_pred = array([labels[y] for y in argmax(y_pred, axis = 1)])
    y_true = array([labels[y] for y in argmax(y_true, axis = 1)])

    conf_matrix = confusion_matrix(y_true, y_pred, labels = labels)

    disp = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = labels)

    disp.plot()

    plt.title(dataset)
    plt.xticks(rotation = 45)
    plt.savefig(f"{SAVE_FIGURES_PATH}/{dataset}_confusion_matrix.png")


plt.show()
