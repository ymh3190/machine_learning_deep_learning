class ConvolutionalNeuralNetwork():
    def convolutional_neural_network() -> None:
        from tensorflow import keras
        keras.layers.Conv2D(10, kernel_size=(3, 3), activation='relu')
        keras.layers.Conv2D(10, kernel_size=(
            3, 3), activation='relu', padding='same')
        keras.layers.MaxPooling2D(2)
        keras.layers.MaxPooling2D(2, strides=2, padding='valid')
