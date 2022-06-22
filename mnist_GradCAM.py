import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2
import pickle
from tqdm import tqdm
from mnist_cnn import MnistClassifier, serialize_write, serialize_read


class GradCam:
    def __init__(self, model, kind):
        self.model = model
        self.kind = kind

    def grad_cam(self, guide=False):
        # GradCAM用に出力を最終CNNマップ(self.last_conv)とsoftmaxとしたモデルを作成(Functional API)
        grad_model = tf.keras.Model([self.model.inputs],\
            [self.model.get_layer(self.model.self.last_layername).output, self.model.output])

        # tapeにself.last_convの出力からout(prediction結果)までの計算を保存
        with tf.GradientTape() as tape:
            # shape: (sample, layercol, layerrow, layerchannel), (sample, outputs)
            conv_outputs, predictions = self.model(self.X_test)
            class_idx = [np.argmax(pred) for pred in predictions]   # shape: (sample,)
            loss = [pred[class_idx] for pred in predictions]        # shape: (sample,)
        grads = tape.gradient(loss, conv_outputs)    # shape: (sample, layercol, layerrow, layerchannel)

        # cast <class 'tensorflow.python.framework.ops.EagerTensor'>
        # to <class 'numpy.ndarray'>
        conv_outputs = conv_outputs.numpy()     # shape: (sample, layercol, layerrow, layerchannel)
        grads = grads.numpy()                   # shape: (sample, layercol, layerrow, layerchannel)

        # global average pooling
        layer_weights = np.mean(grads, axis=(1, 2))

        # apply weights
        cam = np.sum(
            np.array([conv_outputs[:, :, i] * layer_weights[i] for i in range(layer_weights.shape[-1])])
        , axis=0)          # shape: (sample, layercol, layerrow)
        # 1枚の場合こちらでも可
        #cam = np.dot(conv_outputs, layer_weights)

        # guided back-propagation
        forward_guide = (conv_outputs > 0).astype(int)
        backprop_guide = (grads > 0).astype(int)
        guided_grads = grads * forward_guide * backprop_guide





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

    flag = True

    if flag:
        # 重みを平均化して、レイヤーの出力に乗じる
        weights = np.mean(grads, axis=(0, 1))          # 下処理なし
        #weights = np.mean(guided_grads, axis=(0, 1))    # 下処理あり
        cam = np.sum(
            np.array([output[:, :, i] * weights[i] for i in range(weights.shape[-1])])
        , axis=0)
        #cam = np.dot(output, weights)      # 上と同じ(カラー不可)

        guided_grads = np.mean(guided_grads, axis=2)
        # ReLUの代わり
        cam  = np.maximum(cam, 0)
        cam *= guided_grads
        # cam画像を元画像と同じ大きさにスケーリング
        cam = cv2.resize(cam, (28,28), cv2.INTER_LINEAR)

        # camヒートマップを計算(255倍しておく)
        heatmap = cam / cam.max() * 255
        # camモノクロヒートマップに疑似的に色をつける
        jet_cam = cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_HOT)
    else:
        weights = np.mean(grads, axis=(0, 1))    # 下処理なし
        cam = np.sum(
            np.array([output[:, :, i] * weights[i] for i in range(weights.shape[-1])])
        , axis=0)
        #cam = np.dot(output, weights)      # 上と同じ(カラー不可)
        cam  = np.maximum(cam, 0)
        cam = cv2.resize(cam, (28,28), cv2.INTER_LINEAR)
        heatmap = cam / cam.max() * 255
        jet_cam = cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_HOT)

    # もとの画像を白黒反転後カラー化(255倍しておく)
    org_img = cv2.cvtColor(np.uint8(x * 255), cv2.COLOR_GRAY2BGR)
    cv2.imwrite("heatmap.png", heatmap)
    cv2.imwrite("org-img.png", org_img)
    cv2.imwrite("jet_cam.png", jet_cam)

    # 合成
    out = cv2.addWeighted(src1=org_img, alpha=0.4, src2=jet_cam, beta=0.6, gamma=0)

    #output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    cv2.imwrite("Grad-CAM.png", out)

    # 255で割って返す
    return out / 255


def main():
    pass

def main():
    FIG_NO = 5

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
        cam = grad_cam(new_model,x_test[FIG_NO] ,'last_conv') * 255
        #array2img(Path.cwd() / 'Sample.png', x_test[FIG_NO], x_test[FIG_NO].shape, True)
        #array2img(Path.cwd() / 'Grad-CAM.png', cam, cam.shape[:2])
        #cv2.imwrite("Grad-CAM.png", cam)
        #fig_gradcam = plt.figure()
        #plt.imshow(cam)
        #fig_gradcam.savefig("Grad-CAM.png")



if __name__ == '__main__':
    main()
