class ConvolutionalNeuralNetwork():
    def convolutional_neural_network() -> None:
        from tensorflow import keras
        keras.layers.Conv2D(10, kernel_size=(3, 3), activation='relu')
        keras.layers.Conv2D(10, kernel_size=(
            3, 3), activation='relu', padding='same')
        keras.layers.MaxPooling2D(2)
        keras.layers.MaxPooling2D(2, strides=2, padding='valid')

    def convolutional_neural_network_classification() -> None:
        from tensorflow import keras
        from sklearn.model_selection import train_test_split
        (train_input, train_target), (
            test_input, test_target) = keras.datasets.fashion_mnist.load_data()
        train_scaled = train_input.reshape(-1, 28, 28, 1)/255.0
        train_scaled, val_scaled, train_target, val_target = train_test_split(
            train_scaled, train_target, test_size=0.2, random_state=42)
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(32, kernel_size=3,
                  activation='relu', padding='same', input_shape=(28, 28, 1)))
        model.add(keras.layers.MaxPooling2D(2))
        model.add(keras.layers.Conv2D(64, kernel_size=3,
                  activation='relu', padding='same'))
        model.add(keras.layers.MaxPooling2D(2))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100, activation='relu'))
        model.add(keras.layers.Dropout(0.4))
        model.add(keras.layers.Dense(10, activation='softmax'))
        # model.summary()
        # keras.utils.plot_model(model)
        # keras.utils.plot_model(model, show_shapes=True,
        #                        to_file='cnn-architecture.png', dpi=300)
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics='accuracy')
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            'best-cnn-model.h5', save_best_only=True)
        early_stopping_cb = keras.callbacks.EarlyStopping(
            patience=2, restore_best_weights=True)
        history = model.fit(train_scaled,
                            train_target,
                            epochs=20,
                            validation_data=(val_scaled, val_target),
                            callbacks=[checkpoint_cb, early_stopping_cb])
        import matplotlib.pyplot as plt
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # plt.legend(['train', 'val'])
        # plt.show()
        model.evaluate(val_scaled, val_target)
        # plt.imshow(val_scaled[0].reshape(28, 28), cmap='gray_r')
        # plt.show()
        preds = model.predict(val_scaled[0:1])
        # print(preds)
        plt.bar(range(1, 11), preds[0])
        plt.xlabel('class')
        plt.ylabel('prob.')
        plt.show()

        classes = ['티셔츠', '바지', '스웨터', '드레스', '코트',
                   '샌달', '셔츠', '스니커즈', '가방', '앵클 부츠']
        # import numpy as np
        # print(classes[np.argmax(preds)])
        test_scaled = test_input.reshape(-1, 28, 28, 1)/255.0
        model.evaluate(test_scaled, test_target)

    def convolutional_neural_network_virtualization() -> None:
        from tensorflow import keras
        BEST_CNN_MODEL_H5 = 'best-cnn-model.h5'
        model = keras.models.load_model(BEST_CNN_MODEL_H5)
        # print(model.layers)
        conv = model.layers[0]
        # print(conv.weights[0].shape, conv.weights[1].shape)
        conv_weights = conv.weights[0].numpy()
        # print(conv_weights.mean(), conv_weights.std())

        import matplotlib.pyplot as plt
        # plt.hist(conv_weights.reshape(-1, 1))
        # plt.xlabel('weight')
        # plt.ylabel('count')
        # plt.show()

        # fig, axs = plt.subplots(2, 16, figsize=(15, 2))
        # for i in range(2):
        #     for j in range(16):
        #         axs[i, j].imshow(conv_weights[:, :, 0, i*16+j],
        #                             vmin=-0.5, vmax=0.5)
        #         axs[i, j].axis('off')
        # plt.show()

        no_training_model = keras.Sequential()
        no_training_model.add(keras.layers.Conv2D(
            32, kernel_size=3, activation='relu', padding='same', input_shape=(28, 28, 1)))
        no_training_conv = no_training_model.layers[0]
        # print(no_training_conv.weights[0].shape)
        no_training_weights = no_training_conv.weights[0].numpy()
        # print(no_training_weights.mean(), no_training_weights.std())
        # plt.hist(no_training_weights.reshape(-1, 1))
        # plt.xlabel('weight')
        # plt.ylabel('count')
        # plt.show()

        # fig, axs = plt.subplots(2, 16, figsize=(15, 2))
        # for i in range(2):
        #     for j in range(16):
        #         axs[i, j].imshow(no_training_weights[:, :, 0,
        #                          i*16+j], vmin=-0.5, vmax=0.5)
        #         axs[i, j].axis('off')
        # plt.show()

        """함수형 API """
        inputs = keras.Input(shape=(784,))
        dense1 = keras.layers.Dense(100, activation='sigmoid')
        dense2 = keras.layers.Dense(10, activation='softmax')
        hidden = dense1(inputs)
        outputs = dense2(hidden)
        model = keras.Model(inputs, outputs)
        conv_acti = keras.Model(model.input, model.layers[0].output)
        (train_input, train_target), (test_input,
                                      test_target) = keras.datasets.fashion_mnist.load_data()
        # plt.imshow(train_input[0], cmap='gray_r')
        # plt.show()
        inputs = train_input.reshape(-1, 28, 28, 1)/255.0
        feature_maps = conv_acti.predict(inputs)
        # print(feature_maps.shape)
        fig, axs = plt.subplots(4, 8, figsize=(15, 8))
        for i in range(4):
            for j in range(8):
                axs[i, j].imshow(feature_maps[0, :, :, i*8+j])
                axs[i, j].axis('off')
        plt.show()
