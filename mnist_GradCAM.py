import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2
import pickle
from tqdm import tqdm
from mnist_cnn import MnistClassifier, serialize_write, serialize_read


class GradCam:
    def __init__(self, trained_model):
        self.trained_model = trained_model

    def get_cam(self, test_no: int) -> np.ndarray:
        # 対象画像、次元追加(読み込む画像が1枚なため、次元を増やしておかないとmodel.predictが出来ない)
        img = self.trained_model.X_test[test_no]
        img_wide = np.expand_dims(img, axis=0)

        # GradCAM用に出力を最終CNNマップ(self.last_conv)とsoftmaxとしたモデルを作成(Functional API)
        grad_model = tf.keras.Model([self.trained_model.model.inputs],\
            [self.trained_model.model.get_layer(self.trained_model.last_layername).output,
            self.trained_model.model.output])

        # tapeにself.last_convの出力からout(prediction結果)までの計算を保存
        with tf.GradientTape() as tape:
            # shape: (layercol, layerrow, layerchannel), (outputs,)
            conv_outputs, predictions = grad_model(img_wide)
            class_idx = np.argmax(predictions[0])       # shape: (1,)
            loss = predictions[0][class_idx]            # shape: (1,)
        # backpropを取得
        grads = tape.gradient(loss, conv_outputs)       # shape: (layercol, layerrow, layerchannel)

        # cast <class 'tensorflow.python.framework.ops.EagerTensor'>
        # to <class 'numpy.ndarray'>
        conv_outputs = conv_outputs.numpy()[0]     # shape: (layercol, layerrow, layerchannel)
        grads = grads.numpy()[0]                   # shape: (layercol, layerrow, layerchannel)

        # global average pooling
        layer_weights = np.mean(grads, axis=(0, 1))     # shape: (layerchannel,)

        # apply weights
        cam = np.sum(
                        np.array([conv_outputs[:, :, i] * layer_weights[i]
                        for i in range(len(layer_weights))]), axis=0
                    )          # shape: (layercol, layerrow)
        # 1枚の場合こちらでも可
        #cam = np.dot(conv_outputs, layer_weights)          # shape: (layercol, layerrow)

        # reluを通す
        cam_relu = np.maximum(cam, 0)
        # camをリサイズ
        cam_resized = cv2.resize(cam_relu, self.trained_model.input_shape, cv2.INTER_LINEAR) # shape: (layercol, layerrow)

        # make heatmap
        heatmap = cam_resized / cam_resized.max() * 255       # shape: (layercol, layerrow)
        # apply color
        hm_colored = cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_HOT)    # shape: (layercol, layerrow)

        # 元の画像をカラー化
        org_img = cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_GRAY2BGR)     # shape: (layercol, layerrow)

        # 合成
        out = cv2.addWeighted(src1=org_img, alpha=0.4, src2=hm_colored, beta=0.6, gamma=0)     # shape: (layercol, layerrow)

        return out              # shape: (1,)

    def save_img(self, img: np.ndarray, test_no: int, save_dir: Path, extension='png', overwrite=True):
        output_filename = f'{test_no}_cam.{extension}'
        output_filepath = Path.cwd() / save_dir / output_filename
        if overwrite or not output_filepath.exists():
            cv2.imwrite(str(output_filepath), img)
        else:
            print(f'There already exists {output_filepath.name}. Overwrite is not valid.')


def main():
    mnist = MnistClassifier.reconstructmodel()
    cam = GradCam(mnist)

    # 間違えたテストデータをpngに出力
    for i in mnist.index_failure:
        Path(f'failure/{i}').mkdir(exist_ok=True)
        img = cam.get_cam(i)
        cam.save_img(img, i, Path(f'failure/{i}'))

    # 全テストデータをpngに出力
    for i, y_test in enumerate(tqdm(mnist.y_test)):
        Path(f'{y_test}/{i}').mkdir(exist_ok=True)
        img = cam.get_cam(i)
        cam.save_img(img, i, Path(f'{y_test}/{i}'))


if __name__ == '__main__':
    main()
