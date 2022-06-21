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
        """MNISTデータセットをロード
        """
        (self.X_train, self.y_train), (self.X_test, self.y_test) \
            = tf.keras.datasets.mnist.load_data()
        self.X_train = self.X_train / 255
        self.X_test = self.X_test / 255

    def cnnconfig(self):
        """諸元の設定
        """
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
        self.predict = None
        self.index_failure = None

    def makecnnmodel(self):
        """モデルの作成(Sequential API)
        """
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
        """訓練
        """
        self.history = self.model.fit(self.X_train, self.y_train,\
            batch_size=self.batchsize, epochs=self.epochs, validation_split=self.validation_rate)
        self.model.save(self.model_savefile)

    def drawlossgraph(self):
        """損失関数の描画
        """
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
        """テストデータによる評価
        """
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)

    @classmethod
    def deeplearning(cls):
        """ディープラーニングを実行

        Returns:
            mnist: MnistClassifierオブジェクト
        """
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
        """保存済みモデルの再構築

        Raises:
            e: 指定した名前の保存済みモデルがない場合

        Returns:
            mnist: MnistClassifierオブジェクト
        """
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

    def prediction(self) -> tuple:
        """全テストデータで予測、予測結果と間違えたテストデータのインデックスpickle化して保存

        Returns:
            tuple: 予測結果と間違えたテストデータのインデックス
        """
        predictions = self.model.predict(self.X_test)
        predictions = [np.argmax(pred) for pred in predictions]
        index_failure = [i for i, (p, t) in enumerate(zip(predictions, self.y_test)) if p != t]
        serialize_write((predictions, index_failure), Path.cwd() / 'predictions')
        self.predict = predictions
        self.index_failure = index_failure
        return (predictions, index_failure)

    def array2img(self, test_no: int, save_dir: Path, extension='png', target_dpi=None, inverse=False, overwrite=True):
        """テストデータを画像ファイルとして保存

        Args:
            test_no (int): テストデータの何番目か
            save_dir (Path): 保存先ディレクトリ
            extension (str, optional): ファイル拡張子. Defaults to 'png'.
            target_dpi (_type_, optional): リサイズするdpi. Defaults to None.
            inverse (bool, optional): 白黒反転. Defaults to False.
            overwrite (bool, optional): 同名ファイルに上書きするか. Defaults to True.
        """
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
    """pickle化

    Args:
        obj (Any): pickle化するオブジェクト
        filepath (Path): 保存するパス
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def serialize_read(filepath: Path):
    """_summary_

    Args:
        filepath (Path): 読みだすpickleファイル

    Returns:
        Any: 非pickle化されたオブジェクト
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def main():
    # ディープラーニング実行
    #mnist = MnistClassifier.deeplearning()

    # 保存済みモデル再構築
    mnist = MnistClassifier.reconstructmodel()

    # 予測結果、間違えたインデックスを取得
    predictions, index_failure = mnist.prediction()

    # 0 - 9, failureのディレクトリ作成
    cwd = Path.cwd()
    (cwd / 'failure').mkdir(exist_ok=True)
    for i in range(10):
        (cwd / f'{i}').mkdir(exist_ok=True)

    # 間違えたテストデータをpng, csvに出力
    with open('failure.csv', 'w', encoding='utf-8') as f:
        f.write('index,prediction,answer\n')
        for i in index_failure:
            mnist.array2img(i, Path(f'failure'), inverse=True)
            f.write(f'{i},{predictions[i]},{mnist.y_test[i]}\n')

    # 全テストデータをpngに出力
    for i, y_test in enumerate(tqdm(mnist.y_test)):
        mnist.array2img(i, Path(f'{y_test}'), inverse=True)

if __name__ == '__main__':
    main()
