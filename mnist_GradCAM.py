import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mnist_cnn import mnist_show

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

    # 画像を元画像と同じ大きさにスケーリング
    cam = cv2.resize(cam, (28,28), cv2.INTER_LINEAR)
    # ReLUの代わり
    cam  = np.maximum(cam, 0)
    # ヒートマップを計算
    heatmap = cam / cam.max()

    # モノクロ画像に疑似的に色をつける
    jet_cam = cv2.applyColorMap(np.uint8(255.0*heatmap), cv2.COLORMAP_JET)
    # RGBに変換
    rgb_cam = cv2.cvtColor(jet_cam, cv2.COLOR_BGR2RGB)
    # もとの画像に合成
    #rgb_cam = (np.float32(rgb_cam) + np.expand_dims(x, 2) / 2)
    org_img = cv2.cvtColor(np.uint8(cv2.bitwise_not(x)), cv2.COLOR_GRAY2RGB)
    output = cv2.addWeighted(src1=org_img, alpha=0.3, src2=rgb_cam, beta=0.7, gamma=0)

    return output

def main():
    FIG_NO = 42

    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test/255

    try:
        new_model = tf.keras.models.load_model('my_model')
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
        fig_gradcam = plt.figure()
        plt.imshow(grad_cam(new_model,x_test[FIG_NO] ,'conv'))
        fig_gradcam.savefig("Grad-CAM.png")
        mnist_show(x_test[FIG_NO])

if __name__ == '__main__':
    main()
