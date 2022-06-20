import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2

class MnistClassifier:
    def __init__(self):
        RANDOM_SEED = 0
        tf.random.set_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    def LoadMnistDataset(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) \
            = tf.keras.datasets.mnist.load_data()
        self.X_train = self.X_train / 255
        self.X_test = self.X_test / 255

    def CnnConfig(self):
        self.model_name = 'my_model'
        self.model_savefile = 'my_model.h5'
        self.last_layername = "last_conv"
        self.input_shape = (28, 28)
        self.outputs = 10
        self.optimizer = 'Adam'
        self.lossfunc = 'sparse_categorical_crossentropy'
        self.epochs = 10
        self.validation_rate = 0.2
        self.batchsize = 128
        self.train_size = self.y_train.shape[0]
        self.test_size = self.y_test.shape[0]

    def MakeCnnModel(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Reshape(self.input_shape + (1,), input_shape=self.input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'\
                , input_shape=(28, 28)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name=self.last_layername),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.outputs, activation='softmax')
        ], name=self.model_name)
        self.model.summary()
        self.model.compile(optimizer=self.optimizer, loss=self.lossfunc, metrics=['accuracy'])

    def Learn(self):
        self.history = self.model.fit(self.X_train, self.y_train,\
            batch_size=self.batchsize, epochs=self.epochs, validation_split=self.validation_rate)
        self.model.save(self.model_savefile)

    def DrawLossGraph(self):
        loss     = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        nb_epoch = len(loss)
        fig = plt.figure()
        plt.plot(range(nb_epoch), loss,     marker='.', label='loss')
        plt.plot(range(nb_epoch), val_loss, marker='.', label='val_loss')
        plt.legend(loc='best', fontsize=10)
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        print("the graph of losses has been stored as \"Loss.png\"")
        fig.savefig("Loss.png")

    def TestEvaluate(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)

    def Array2Img(self, test_no: int, output_dir: Path, extension='png', target_dpi=None, inverse=False, overwrite=True):
        output_filename = f'{test_no}_out.{extension}'
        output_filepath = output_dir / output_filename
        if overwrite or not output_filepath.exists():
            arr = 1 - arr if inverse else arr
            resized_img = cv2.resize(arr, target_dpi) * 255
            cv2.imwrite(str(output_filepath), resized_img)
        else:
            print(f'There already exists {output_filepath.name}. Overwrite is not valid.')


def main():
    mnist = MnistClassifier()
    mnist.LoadMnistDataset()
    mnist.CnnConfig()
    mnist.MakeCnnModel()
    mnist.Learn()
    mnist.DrawLossGraph()
    mnist.TestEvaluate()




# save a flatten vector as a png
def array2img(output_file_path: Path, arr: np.ndarray, target_dpi: tuple, inverse=False, overwrite=True):
    if overwrite or not output_file_path.exists():
        if inverse:
            arr = 1 - arr
        resized_img = cv2.resize(arr, target_dpi) * 255
        cv2.imwrite(str(output_file_path), resized_img)
    else:
        print(f'There already exists {output_file_path.name}. Overwrite is not valid.')

def _main():
    RANDOM_SEED = 0
    FIG_NO = 0      # an index number of the image saved as a png file and showed
    tf.random.set_seed(RANDOM_SEED)

    # load test data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train/255
    x_test = x_test/255

    # create a NN model
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),     # redundant but needed
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'\
            , input_shape=(28, 28)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name="conv"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ], name='my_model')
    model.summary()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # leaning
    history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.2)
    model.save("my_model.h5")

    # evaluation
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    # prediction
    predictions = model.predict(x_test)
    print(f"the prediction is {predictions[FIG_NO].argmax()}.")
    print(f"the answer is {y_test[FIG_NO]}.")
    print("the input image has been stored as \"Sample.png\"")
    array2img(Path.cwd() / 'Sample.png', x_test[FIG_NO], (28, 28))

    # show the result of prediction
    loss     = history.history['loss']
    val_loss = history.history['val_loss']
    nb_epoch = len(loss)
    fig = plt.figure()
    plt.plot(range(nb_epoch), loss,     marker='.', label='loss')
    plt.plot(range(nb_epoch), val_loss, marker='.', label='val_loss')
    plt.legend(loc='best', fontsize=10)
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    #plt.show()
    print("the graph of losses has been stored as \"Loss.png\"")
    fig.savefig("Loss.png")


if __name__ == '__main__':
    main()
