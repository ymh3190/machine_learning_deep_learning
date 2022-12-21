class ConvolutionalNeuralNetwork():
    def convolutional_neural_network() -> None:
        from tensorflow import keras
        # keras.layers.Conv2D(10, kernel_size=(3, 3), activation='relu')
        # keras.layers.Conv2D(10, kernel_size=(
        #     3, 3), activation='relu', padding='same')
        # keras.layers.MaxPooling2D(2)
        # keras.layers.MaxPooling2D(2, strides=2, padding='valid')
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
        keras.utils.plot_model(model)
        keras.utils.plot_model(model, show_shapes=True,
                               to_file='cnn-architecture.png', dpi=300)
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
        import numpy as np
        # print(classes[np.argmax(preds)])
        test_scaled = test_input.reshape(-1, 28, 28, 1)/255.0
        model.evaluate(test_scaled, test_target)

        def side() -> None:
            from PIL import Image
            import numpy as np
            import os
            import matplotlib.pyplot as plt
            imgs = np.array([np.array(Image.open(f'static/{img}'))/255.0 for img in os.listdir(
                'static') if img.endswith('.jpg')])
            plt.show()
