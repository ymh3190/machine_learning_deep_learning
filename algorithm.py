from classification import Classification
from regression import Regression


def algorithm() -> None:
    # Classification.classify()
    # Classification.logistic_regression()
    # Regression.regress()
    # Regression.regress_linear()
    # Regression.regress_linear_multiple()
    # Classification.stochastic_gradient_descent()
    # Classification.classify_decision_tree()
    # Classification.classify_cross_validation()
    # Classification.classify_tree_ensemble()
    # Classification.classify_gradient_boosting()
    # Classification.classify_histogram_based_gradient_boosting()

    def clustering() -> None:
        import os
        FRUITS_300_NPY = 'fruits_300.npy'
        if FRUITS_300_NPY in os.listdir(os.getcwd()):
            pass
        else:
            import wget
            wget.download('https://bit.ly/fruits_300_data',
                          out=FRUITS_300_NPY)

        import numpy as np
        import matplotlib.pyplot as plt
        fruits = np.load(FRUITS_300_NPY)
        # print(fruits.shape)
        # print(fruits[0, 0, :])
        # plt.imshow(fruits[0], cmap='gray')
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(fruits[100], cmap='gray_r')
        # axs[1].imshow(fruits[200], cmap='gray_r')
        # plt.show()

        apple = fruits[0:100].reshape(-1, 100*100)
        pineapple = fruits[100:200].reshape(-1, 100*100)
        banana = fruits[200:300].reshape(-1, 100*100)
        """ axis=0이면 행, 1이면 열 """
        # print(apple.mean(axis=1))
        """ 샘플별 평균 비교 """
        # plt.hist(np.mean(apple, axis=1), alpha=0.8)
        # plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
        # plt.hist(np.mean(banana, axis=1), alpha=0.8)
        # plt.legend(['apple', 'pineapple', 'banana'])
        # plt.show()
        """ 픽셀별 평균 비교 """
        # fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        # axs[0].bar(range(10000), np.mean(apple, axis=0))
        # axs[1].bar(range(10000), np.mean(pineapple, axis=0))
        # axs[2].bar(range(10000), np.mean(banana, axis=0))
        # plt.show()
        apple_mean = np.mean(apple, axis=0).reshape(100, 100)
        pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
        banana_mean = np.mean(banana, axis=0).reshape(100, 100)
        fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        axs[0].imshow(apple_mean, cmap='gray_r')
        axs[1].imshow(pineapple_mean, cmap='gray_r')
        axs[2].imshow(banana_mean, cmap='gray_r')
        plt.show()

    clustering()
