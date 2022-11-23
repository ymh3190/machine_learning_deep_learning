from feature import *


def answer_p64() -> None:
    """정확도가 1이하로 내려갈 때의 값 찾기
    """
    length = bream_length+smelt_length
    weight = bream_weight+smelt_weight
    fish_data = [[l, w] for l, w in zip(length, weight)]
    fish_target = [1]*35 + [0]*14
    from sklearn.neighbors import KNeighborsClassifier
    kn = KNeighborsClassifier()
    kn.fit(fish_data, fish_target)

    for n in range(5, 50):
        kn.n_neighbors = n
        score = kn.score(fish_data, fish_target)
        if score < 1:
            print(n, score)
            break


def answer_p128() -> None:
    """과대적합과 과소적합에 대한 이해 향상
    """
    from sklearn.model_selection import train_test_split
    train_input, test_input, train_target, test_target = train_test_split(
        perch_length, perch_weight, random_state=42)
    from sklearn.neighbors import KNeighborsRegressor
    import numpy as np
    import matplotlib.pyplot as plt
    knr = KNeighborsRegressor()
    x = np.arange(5, 46).reshape(-1, 1)
    train_input = train_input.reshape(-1, 1)
    for n in [1, 5, 10]:
        knr.n_neighbors = n
        knr.fit(train_input, train_target)
        prediction = knr.predict(x)
        plt.scatter(train_input, train_target)
        plt.plot(x, prediction, color='green')
        plt.title(f'n_neighbors = {n}')
        plt.xlabel('length')
        plt.ylabel('weight')
        plt.show()
