import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2
import pickle
from tqdm import tqdm

class MnistClassifier:
    def __init__(self):
        RANDOM_SEED = 0
        tf.random.set_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    def loadmnistdataset(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) \
            = tf.keras.datasets.mnist.load_data()
        self.X_train = self.X_train / 255
        self.X_test = self.X_test / 255

    def cnnconfig(self):
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

    def makecnnmodel(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Reshape(self.input_shape + (1,), input_shape=self.input_shape),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu'\
                , input_shape=(28, 28)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', name=self.last_layername),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.outputs, activation='softmax')
        ], name=self.model_name)
        self.model.summary()
        self.model.compile(optimizer=self.optimizer, loss=self.lossfunc, metrics=['accuracy'])

    def training(self):
        self.history = self.model.fit(self.X_train, self.y_train,\
            batch_size=self.batchsize, epochs=self.epochs, validation_split=self.validation_rate)
        self.model.save(self.model_savefile)

    def drawlossgraph(self):
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

    def testevaluate(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)

    @classmethod
    def deeplearning(cls):
        mnist = cls()
        mnist.loadmnistdataset()
        mnist.cnnconfig()
        mnist.makecnnmodel()
        mnist.training()
        mnist.drawlossgraph()
        mnist.testevaluate()
        return mnist

    @classmethod
    def reconstructmodel(cls):
        mnist = cls()
        mnist.loadmnistdataset()
        mnist.cnnconfig()
        try:
            mnist.model = tf.keras.models.load_model(mnist.model_savefile)
        except OSError as e:
            print("No model exists")
            raise e
        else:
            mnist.model.summary()
            mnist.testevaluate()
        return mnist

    def array2img(self, test_no: int, save_dir: Path, extension='png', target_dpi=None, inverse=False, overwrite=True):
        output_filename = f'{test_no}.{extension}'
        output_filepath = Path.cwd() / save_dir / output_filename
        target_dpi = self.input_shape if target_dpi == None else target_dpi
        if overwrite or not output_filepath.exists():
            arr = 1 - self.X_test[test_no] if inverse else self.X_test[test_no]
            resized_img = cv2.resize(arr, target_dpi) * 255
            cv2.imwrite(str(output_filepath), resized_img)
        else:
            print(f'There already exists {output_filepath.name}. Overwrite is not valid.')

def serialize_write(obj, filepath: Path):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
def serialize_read(filepath: Path):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def main():
    #mnist = MnistClassifier.deeplearning()
    mnist = MnistClassifier.reconstructmodel()
    for i in range(10):
        cwd = Path.cwd()
        (cwd / f'{i}').mkdir(exist_ok=True)
    for i, y_test in enumerate(tqdm(mnist.y_test)):
        mnist.array2img(i, Path(f'{y_test}'), inverse=True)

if __name__ == '__main__':
    main()
