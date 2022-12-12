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
    from artificial_neural_network import ArtificialNeuralNetwork
    ArtificialNeuralNetwork.artificial_neural_network()


def algorithm() -> None:
    # classify()
    # regress()
    # cluster()
    artificial_neural_network()
