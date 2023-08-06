import tensorflow as tf
import tensorflow_text as text
import tensorflow_addons as tfa
from sklearn.metrics import f1_score
from numpy import argmax, array
import argparse

from load_data import load_data


def main(args):
    model = tf.keras.models.load_model(args.model)
    
    _, _, X_test, _, _, y_test, labels = load_data(args.data, args.mode)
    y_pred = model.predict(X_test)

    if args.mode == 'multi-class':
        y_pred = array([labels[y] for y in argmax(y_pred, axis = 1)])
        y_test = array([labels[y] for y in argmax(y_test, axis = 1)])

    else:
        y_pred = [round(y[0]) for y in y_pred]

    print(f"Macro F1: {f1_score(y_test, y_pred, average = 'macro')*100:.2f}%")
    print(f"Weighted F1: {f1_score(y_test, y_pred, average = 'weighted')*100:.2f}%")
    print(f"Micro F1: {f1_score(y_test, y_pred, average = 'micro')*100:.2f}%")


if __name__ == "__main__":
      parser = argparse.ArgumentParser('Testing BERT on SUD classification')

      parser.add_argument('--data', type = str, help = 'Path to the dataset')
      parser.add_argument('--model', type = str, help = 'Path to the trained model')
      parser.add_argument('--mode', type = str, default = 'binary', choices = ['binary', 'multi-class'], help = 'Perform binary or multi-class SUD classification')

      args = parser.parse_args()
      main(args)