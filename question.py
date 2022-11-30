from feature import *


def answer(arg) -> None:
    if arg == 0:
        answer_p64()
    elif arg == 1:
        answer_p128()
    elif arg == 2:
        answer_p241()


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


def answer_p241() -> None:
    """결정트리
    """
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    import pandas as pd
    from sklearn.model_selection import train_test_split
    wine = pd.read_csv('https://bit.ly/wine_csv_data')
    data = wine[['alcohol', 'sugar',  'pH']].to_numpy()
    target = wine['class'].to_numpy()
    train_input, test_input, train_target, test_target = train_test_split(
        data, target)
    dt = DecisionTreeClassifier(
        max_depth=3, min_impurity_decrease=0.0005, random_state=42)
    dt.fit(train_input, train_target)
    print(dt.score(train_input, train_target))
    print(dt.score(test_input, test_target))
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 15))
    plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar',  'pH'])
    plt.show()
