import tensorflow as tf
import matplotlib.pyplot as plt

# save a flatten vector as a png
def mnist_show(flattenpixel):
    """
    Args: 
        flattenpixel: a flatten vector of one image
    Returns:
        None
    """

    fig = plt.figure()
    plt.imshow(flattenpixel.reshape(28, 28), cmap='Greys')
    #plt.show()
    fig.savefig("Sample.png")


def main():
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
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name="conv"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ], name='my_model')
    model.summary()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # leaning
    history = model.fit(x_train, y_train, batch_size=128, epochs=20, validation_split=0.2)
    model.save("my_model")

    # evaluation
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    # prediction
    predictions = model.predict(x_test)
    print(f"the prediction is {predictions[FIG_NO].argmax()}.")
    print(f"the answer is {y_test[FIG_NO]}.")
    print("the input image has been stored as \"Sample.png\"")
    mnist_show(x_test[FIG_NO])

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
