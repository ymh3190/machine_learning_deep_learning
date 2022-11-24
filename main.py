from feature import *
import scikit_learn
import question


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
    # scikit_learn.classify()
    # scikit_learn.regress()
    # question.answer_p64()
    # question.answer_p128()
    # scikit_learn.regress_linear()
    # scikit_learn.regress_linear_multiple()
    scikit_learn.regress_logistic()


if __name__ == '__main__':
    main()
