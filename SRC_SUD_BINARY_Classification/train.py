import tensorflow as tf
import tensorflow_addons as tfa
from os import listdir
from BERT_models import build_classifier_model
from load_data import split_binary_data


DATASET_TRAIN_PATH      = "PATH_TO_THE_DATA"
BATCH_SIZE              = 32
EPOCHS                  = 5
LEARNING_RATE           = 3e-6            #2e-5
TEST                    = True
SAVE_MODEL              = True


X_train, X_validation, X_test, y_train, y_validation, y_test = split_binary_data(DATASET_TRAIN_PATH)

print(f"Training on {len(X_train)} samples\nUsing {len(X_validation)} samples for validation and {len(X_test)} for testing")

classifier_model = build_classifier_model('bert_en_uncased_L-12_H-768_A-12')
classifier_model.summary()

metrics = [tfa.metrics.F1Score(num_classes = 1, threshold = 0.5, average = "micro", name = 'f1_micro'),
          tf.metrics.BinaryAccuracy()]

loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.1)

optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE)

classifier_model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

history = classifier_model.fit(x = X_train,
                               y = y_train,
                               batch_size = BATCH_SIZE,
                               validation_data = (X_validation, y_validation),
                               epochs = EPOCHS)


if TEST:
      test_result = classifier_model.evaluate(x = X_test, y = y_test, return_dict = True)
      print(f'Test Result: {test_result}')

if SAVE_MODEL:
      classifier_model.save(f'./saved_models/0{len(listdir("saved_models")) + 1}', include_optimizer=False)



# plot_training_curves(history)
