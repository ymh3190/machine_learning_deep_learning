def classify() -> None:
    from classification import Classification
    Classification.classify()
    Classification.classify_logistic_regression()
    Classification.classify_stochastic_gradient_descent()
    Classification.classify_decision_tree()
    Classification.classify_cross_validation()
    Classification.classify_tree_ensemble()
    Classification.classify_gradient_boosting()
    Classification.classify_histogram_based_gradient_boosting()


def regress() -> None:
    from regression import Regression
    Regression.regress()
    Regression.regress_linear()
    Regression.regress_linear_multiple()


def cluster() -> None:
    from cluster import Cluster
    Cluster.cluster()
    Cluster.cluster_center()
    Cluster.dimension_reduction()


def artificial_neural_network() -> None:
    from ann import ArtificialNeuralNetwork as ann
    ann.artificial_neural_network()
    ann.deep_neural_network()
    ann.train_neural_network()


def convolutional_neural_network() -> None:
    from cnn import ConvolutionalNeuralNetwork as cnn
    # cnn.convolutional_neural_network()
    # cnn.convolutional_neural_network_classification()
    cnn.convolutional_neural_network_virtualization()


def algorithm() -> None:
    # classify()
    # regress()
    # cluster()
    # artificial_neural_network()
    convolutional_neural_network()
