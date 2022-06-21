import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2
import pickle
from tqdm import tqdm
from mnist_cnn import MnistClassifier, serialize_write, serialize_read


class GradCam:
    def __init__(self, model):
        self.model = model

    def grad_cam(self):
        # GradCAM用に出力を最終CNNマップ(self.last_conv)とsoftmaxとしたモデルを作成(Functional API)
        grad_model = tf.keras.Model([self.model.inputs],\
            [self.model.get_layer(self.model.self.last_layername).output, self.model.output])

        # tapeにself.last_convの出力からout(prediction結果)までの計算を保存
        with tf.GradientTape() as tape:
            # 1つだけNoneだとNoneのところで自動でshapeを合わせる、実質Noneにはlen(self.y_test)が入る
            conv_outputs, predictions = self.model(self.X_test)
            class_idx = [np.argmax(pred) for pred in predictions]
            loss = [pred[class_idx] for pred in predictions]




def grad_cam(input_model, x, layer_name):
    """
    Args:
        input_model(object): モデルオブジェクト
        x(ndarray): 画像
        layer_name(string): 畳み込み層の名前
    Returns:
        output_image(ndarray): 元の画像に色付けした画像
    """

    # 画像の前処理
    # 読み込む画像が1枚なため、次元を増やしておかないとmodel.predictが出来ない
    preprocessed_input = np.expand_dims(x, axis=0)

    # 入力1つ、出力2つ(複数ある時はリストで渡す)のfunctional API
    grad_model = tf.keras.models.Model([input_model.inputs],\
    [input_model.get_layer(layer_name).output, input_model.output])

    # tapeにconvの出力からout(prediction結果)までの計算を保存
    # conv_outputs->最後のconv層の出力(None, 24, 24, 64)、predictions->最終出力(None, 10)
    with tf.GradientTape() as tape:
        conv_outputs, prediction = grad_model(preprocessed_input)
        # predictをつけないとtf型のまま　つけるとnumpy
        # tf型の場合後ろにshapeとdtypeがつくがindexは変わらず参照できる
        # ひとつ次元が増える
        class_idx = np.argmax(prediction[0])    # [0]で次元を落とす
        loss = prediction[0][class_idx]     # [0]で次元を落とす
        # loss = prediction[:, class_idx]と書いてもよい

    # 勾配を計算
    output = conv_outputs[0].numpy()    # [0]で次元を落とす
    grads = tape.gradient(loss, conv_outputs)[0].numpy()    # [0]で次元を落とす
    # 保存しておいたtapeからbackpropagationを取得

    gate_f = tf.cast(output > 0, 'float32').numpy()
    gate_r = tf.cast(grads > 0, 'float32').numpy()
    guided_grads = (gate_f * gate_r * grads)

    # 重みを平均化して、レイヤーの出力に乗じる
    weights = np.mean(grads, axis=(0, 1))          # 下処理なし
    weights = np.mean(guided_grads, axis=(0, 1))    # 下処理あり
    cam = np.dot(output, weights)

    # cam画像を元画像と同じ大きさにスケーリング
    cam = cv2.resize(cam, (28,28), cv2.INTER_LINEAR)
    # ReLUの代わり
    cam  = np.maximum(cam, 0)
    # camヒートマップを計算(255倍しておく)
    heatmap = cam / cam.max() * 255
    # camモノクロヒートマップに疑似的に色をつける
    jet_cam = cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_JET)

    # もとの画像を白黒反転後カラー化(255倍しておく)
    org_img = cv2.cvtColor(np.uint8((1 - x) * 255), cv2.COLOR_GRAY2BGR)

    # 合成
    output = cv2.addWeighted(src1=org_img, alpha=0.4, src2=jet_cam, beta=0.6, gamma=0)

    # 255で割って返す
    return output / 255


def main():
    pass

def main():
    FIG_NO = 0

    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test/255

    try:
        new_model = tf.keras.models.load_model('my_model.h5')
    except OSError:
        print("No model exists")
    else:
        new_model.summary()
        test_loss, test_acc = new_model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)
        predictions = new_model.predict(x_test)
        print(f"the prediction is {predictions[FIG_NO].argmax()}.")
        print(f"the answer is {y_test[FIG_NO]}.")
        print("the input image has been stored as \"Grad-CAM.png\"")
        print("the input image has been stored as \"Sample.png\"")
        cam = grad_cam(new_model,x_test[FIG_NO] ,'last_conv')
        #array2img(Path.cwd() / 'Sample.png', x_test[FIG_NO], x_test[FIG_NO].shape, True)
        #array2img(Path.cwd() / 'Grad-CAM.png', cam, cam.shape[:2])



if __name__ == '__main__':
    main()
