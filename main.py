from feature import *
from classification import Classification
from regression import Regression
import question

"""
지도 학습 알고리즘은 크게 분류와 회귀로 나눕니다.
분류는 말 그대로 샘플을 몇 개의 클래스 중 하나로 분류하는 것
회귀는 임의의 어떤 숫자를 예측하는 문제
"""


def show_plt() -> None:
    """그래프 그리기
    """
    import matplotlib.pyplot as plt
    plt.scatter(bream_length, bream_weight)
    plt.scatter(smelt_length, smelt_weight)
    plt.xlabel('length')
    plt.ylabel('weight')
    plt.show()


def main():
    # show_plt()
    # Classification.classify()
    # Classification.regress_logistic()
    # question.answer_p64()
    # question.answer_p128()
    # Regression.regress()
    # Regression.regress_linear()
    # Regression.regress_linear_multiple()
    # Classification.stochastic_gradient_descent()
    Classification.decision_tree()


if __name__ == '__main__':
    main()
