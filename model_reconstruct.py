import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from main import mnist_show

def main():
    FIG_NO = 5

    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test/255

    try:
        new_model = tf.keras.models.load_model('my_model')
        new_model.summary()
        test_loss, test_acc = new_model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)
        predictions = new_model.predict(x_test)
        print(f"the prediction is {predictions[FIG_NO].argmax()}.")
        print(f"the answer is {y_test[FIG_NO]}.")
        print("the input image has been stored as \"Sample.png\"")
        mnist_show(x_test[FIG_NO])
    except:
        print("No model exists")

if __name__ == '__main__':
    main()
