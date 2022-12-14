class ArtificialNeuralNetwork():
    def artificial_neural_network() -> None:
        from tensorflow import keras
        (train_input, train_target), (
            test_input, test_target) = keras.datasets.fashion_mnist.load_data()
        # print(train_input.shape, train_target.shape)
        # print(test_input.shape, test_target.shape)
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 10, figsize=(10, 10))
        for i in range(10):
            axs[i].imshow(train_input[i], cmap='gray_r')
            axs[i].axis('off')
        # print([train_target[i] for i in range(10)])
        # plt.show()
        import numpy as np
        # print(np.unique(train_target, return_counts=True))
        train_scaled = train_input/255.0
        train_scaled = train_scaled.reshape(-1, 28*28)
        # print(train_scaled.shape)
        from sklearn.model_selection import cross_validate
        from sklearn.linear_model import SGDClassifier
        # sc = SGDClassifier(loss='log_loss', max_iter=5, random_state=42)
        # scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
        # print(np.mean(scores['test_score']))
        import tensorflow as tf
        from tensorflow import keras
        from sklearn.model_selection import train_test_split
        train_scaled, val_scaled, train_target, val_target = train_test_split(
            train_scaled, train_target, test_size=0.2, random_state=42)
        # print(train_scaled.shape, train_target.shape)
        dense = keras.layers.Dense(
            10, activation='softmax', input_shape=(784,))
        model = keras.Sequential(dense)
        model.compile(loss='sparse_categorical_crossentropy',
                      metrics='accuracy')
        # print(train_target[:10])
        model.fit(train_scaled, train_target, epochs=5)
        model.evaluate(val_scaled, val_target)

    def deep_neural_network() -> None:
        """층을 더 추가할 수 있다 그래서 deep learning"""
        from tensorflow import keras
        (train_input, train_target), (
            test_input, test_target) = keras.datasets.fashion_mnist.load_data()
        from sklearn.model_selection import train_test_split
        train_scaled = train_input/255.0
        train_scaled = train_scaled.reshape(-1, 28*28)
        train_scaled, val_scaled, train_target, val_target = train_test_split(
            train_scaled, train_target, test_size=0.2, random_state=42)
        """dense 추가 방법 1"""
        dense1 = keras.layers.Dense(
            100, activation='sigmoid', input_shape=(784,))
        dense2 = keras.layers.Dense(10, activation='softmax')
        """출력층을 마지막에 배치"""
        model = keras.Sequential([dense1, dense2])
        # model.summary()
        """dense 추가 방법 2"""
        model = keras.Sequential([
            keras.layers.Dense(100, activation='sigmoid', input_shape=(784,),
                               name='hidden'),
            keras.layers.Dense(10, activation='softmax', name='output')
        ], name='Fashion MNIST Model')
        # model.summary()
        # model = keras.Sequential()
        # model.add(keras.layers.Dense(
        #     100, activation='sigmoid', input_shape=(784,)))
        # model.add(keras.layers.Dense(10, activation='softmax'))
        # model.summary()
        # model.compile(loss='sparse_categorical_crossentropy',
        #               metrics='accuracy')
        # model.fit(train_scaled, train_target, epochs=5)

        """Flatten class"""
        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(28, 28)))
        model.add(keras.layers.Dense(100, activation='relu'))
        model.add(keras.layers.Dense(10, activation='softmax'))
        # model.summary()
        (train_input, train_target), (
            test_input, test_target) = keras.datasets.fashion_mnist.load_data()
        from sklearn.model_selection import train_test_split
        train_scaled = train_input/255.0
        train_scaled, val_scaled, train_target, val_target = train_test_split(
            train_scaled, train_target, test_size=0.2, random_state=42)
        model.compile(loss='sparse_categorical_crossentropy',
                      metrics='accuracy')
        model.fit(train_scaled, train_target, epochs=5)
        model.evaluate(val_scaled, val_target)
