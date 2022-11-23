from feature import *

"""
지도 학습 알고리즘은 크게 분류와 회귀로 나눕니다.
분류는 말 그대로 샘플을 몇 개의 클래스 중 하나로 분류하는 것
회귀는 임의의 어떤 숫자를 예측하는 문제
"""


def classify() -> None:
    """첫 번째 머신 러닝 프로그램 : k-최근접 이웃(k-Nearest Neighbors) 알고리즘
    - 어떤 데이터에 대한 답을 구할 때 주위의 다른 데이터를 보고 다수를 차지하는 것을 정답으로 합니다.
    - 단점 : 새로운 데이터에 대해 예측할 때는 가장 가까운 직선거리에 어떤 데이터가 있는지를 살핀다 \
        이러한 특징으로 데이터가 아주 많은 경우 사용하기 어려움
    - 사례 기반 학습
    """
    length = bream_length+smelt_length
    weight = bream_weight+smelt_weight
    fish_data = [[l, w] for l, w in zip(length, weight)]
    fish_target = [1]*35 + [0]*14
    from sklearn.neighbors import KNeighborsClassifier
    kn = KNeighborsClassifier()
    # training
    kn.fit(fish_data, fish_target)
    # 평가
    print(kn.score(fish_data, fish_target))
    # 예측
    print(kn.predict([[30, 600]]))
    # 매개변수 _fit_X = fish_data
    print(kn._fit_X)
    # 매개변수 _y = fish_target
    print(kn._y)
    # 기본 참고 데이터 개수 5개
    kn49 = KNeighborsClassifier(n_neighbors=49)
    kn49.fit(fish_data, fish_target)
    print(kn49.score(fish_data, fish_target))
    # 슬라이싱
    print(fish_data[4])
    print(fish_data[:5])
    print(fish_data[44:])
    train_input = fish_data[:35]
    train_target = fish_target[:35]
    test_input = fish_data[35:]
    test_target = fish_target[35:]
    kn = kn.fit(train_input, train_target)
    # 샘플링 편향(sampling bias)으로 인해 평가결과 0점
    print(kn.score(test_input, test_target))
    # 샘플링 편향 문제를 해결하기 위한 numpy(넘파이)
    # 리스트를 넘파이 배열로 변경
    import numpy as np
    input_arr = np.array(fish_data)
    target_arr = np.array(fish_target)
    print(input_arr)
    # 배열의 크기를 알려주는 shape 속성(샘플 수, feature 수)
    print(input_arr.shape)
    # 42는 넘파이 난수 생성에서 흔히 쓰임
    np.random.seed(42)
    # 0~48까지 1씩 증가하는 리스트를 만듦
    index = np.arange(49)
    # 주어진 배열을 무작위로 섞는다
    np.random.shuffle(index)
    print(index)
    # array indexing(배열 인덱싱) : 여러 개의 원소 선택
    print(input_arr[[1, 3]])
    train_input = input_arr[index[:35]]
    train_target = target_arr[index[:35]]
    print(input_arr[13], train_input[0])
    test_input = input_arr[index[35:]]
    test_target = target_arr[index[35:]]
    import matplotlib.pyplot as plt
    # 넘파이 2차원 배열은 행과 열 인덱스를 콤마로 나누어 지정할 수 있다
    # plt.scatter(train_input[:, 0], train_input[:, 1])
    # plt.scatter(test_input[:, 0], test_input[:, 1])
    plt.xlabel('length')
    plt.ylabel('weight')
    # plt.show()
    print('--------------------------------')
    print(kn.score(test_input, test_target))
    kn = kn.fit(train_input, train_target)
    # numpy 배열 반환, 파이썬 리스트 아님
    print(kn.predict(test_input))
    print(test_target)
    # 파이썬 리스트 순회없이 넘파이를 이용한 데이터 구성
    np.column_stack(([1, 2, 3, ], [4, 5, 6]))
    fish_data = np.column_stack((fish_length, fish_weight))
    print('--------------------------------')
    print(fish_data[:5])
    # np.ones()와 np.zeros()
    print(np.ones(5))
    # concatenate()
    fish_target = np.concatenate((np.ones(35), np.zeros(14)))
    print(fish_target)
    # 사이킷런으로 훈련 세트와 테스트 세트 나누기
    print('--------------------------------')
    from sklearn.model_selection import train_test_split
    # 기본적으로 25%를 테스트 세트로 분리
    train_input, test_input, train_target, test_target = train_test_split(
        fish_data, fish_target, random_state=42)
    print(train_input.shape, test_input.shape)
    print(train_target.shape, test_target.shape)
    # 개수가 적을 경우 비율이 맞지 않을 수 있다
    print(test_target)
    # stratify 매개변수로 조정 가능
    train_input, test_input, train_target, test_target = train_test_split(
        fish_data, fish_target, stratify=fish_target, random_state=42)

    print(test_target)
    kn.fit(train_input, train_target)
    print(kn.score(test_input, test_target))
    print(kn.predict([[25, 150]]))
    # 1이 아닌 0으로 예측 왜?
    # plt.scatter(train_input[:, 0], train_input[:, 1])
    # plt.scatter(25, 150, marker='^')
    # plt.show()
    # 주어진 샘플에서 가장 가까운 이웃을 찾는 kneighbors(), 이웃까지의 거리와 이웃 샘플 인덱스 반환
    # n_neighbors의 기본값은 5이므로 5개 이웃을 반환
    # distances, indexes = kn.kneighbors([[25, 150]])
    # plt.scatter(train_input[:, 0], train_input[:, 1])
    # plt.scatter(25, 150, marker='^')
    # plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
    # plt.show()
    # print(train_input[indexes])
    # print(train_target[indexes])
    # print(distances)
    # x축 범위 지정
    # plt.xlim((0, 1000))
    # plt.show()
    # 데이터 전처리 : 데이터를 표현하는 기준을 맞춤(인치, 센티 등)
    # 표준점수(standard score) : 가장 널리 사용하는 전처리 방법 중 하나
    # mean() : 평균 계산
    # std() : 표준편차 계산
    # axis=0 : 열 계산, axis=1 행 계산
    mean = np.mean(train_input, axis=0)
    std = np.std(train_input, axis=0)
    print('--------------------------------')
    # print(mean, std)
    # 브로드캐스팅 : 넘파이 배열 사이에서 일어나는 연산
    train_scaled = (train_input - mean)/std
    print(train_scaled)
    new = ([25, 150]-mean)/std
    plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
    plt.scatter(new[0], new[1], marker='^')
    # plt.show()
    kn.fit(train_scaled, train_target)
    test_scaled = (test_input-mean)/std
    print(kn.score(test_scaled, test_target))
    print(kn.predict([new]))
    distances, indexes = kn.kneighbors([new])
    plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
    plt.scatter(new[0], new[1], marker='^')
    plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D')
    plt.show()


def regress() -> None:
    import matplotlib.pyplot as plt
    # plt.scatter(perch_length, perch_weight)
    plt.xlabel('length')
    plt.ylabel('weight')
    # plt.show()
    from sklearn.model_selection import train_test_split
    train_input, test_input, train_target, test_target = train_test_split(
        perch_length, perch_weight, random_state=42)
    # 넘파이 배열 크기는 튜플로 표현된다
    print(np.array([1, 2, 3, 4]).shape)
    # 1개의 특성으로 2차원 배열 만들기
    print(np.array([1, 2, 3, 4]).reshape(2, 2))
    # 크기 2, 피처 2
    print(np.array([1, 2, 3, 4]).reshape(2, 2).shape)
    # 배열의 전체 원소 개수를 매번 외울 필요가 없다
    train_input = train_input.reshape(-1, 1)
    test_input = test_input.reshape(-1, 1)
    print(train_input.shape, test_input.shape)
    print(train_input)
    from sklearn.neighbors import KNeighborsRegressor
    knr = KNeighborsRegressor()
    knr.fit(train_input, train_target)
    print(knr.score(test_input, test_target))
    # 결정계수(R^2) : coefficient of determination
    from sklearn.metrics import mean_absolute_error
    test_prediction = knr.predict(test_input)
    print(mean_absolute_error(test_target, test_prediction))
    # 과대적합(overfitting) : train > test
    # 과소적합(underfitting) : test > train 혹은 두 점수 모두 너무 낮거나
    print(knr.score(train_input, train_target))
    # test : 0.99 > train : 0.96 -> 과소적합 -> 모델을 조금 더 복잡하게 만듦 -> 이웃개수 줄이기
    knr.n_neighbors = 3
    knr.fit(train_input, train_target)
    print(knr.score(train_input, train_target))
    print(knr.score(test_input, test_target))


def regress_linear() -> None:
    """최적의 모델 파라미터(가중치, 절편 등)를 찾는 모델 기반 학습
    """
    from sklearn.model_selection import train_test_split
    train_input, test_input, train_target, test_target = train_test_split(
        perch_length, perch_weight, random_state=42)
    train_input = train_input.reshape(-1, 1)
    test_input = test_input.reshape(-1, 1)
    from sklearn.neighbors import KNeighborsRegressor
    knr = KNeighborsRegressor(n_neighbors=3)
    knr.fit(train_input, train_target)
    # print(knr.predict([[50]]))
    import matplotlib.pyplot as plt
    distances, indexes = knr.kneighbors([[50]])
    # plt.scatter(train_input, train_target)
    # plt.scatter(train_input[indexes], train_target[indexes], marker='D')
    # plt.scatter(50, 1033, marker='^')
    plt.xlabel('length')
    plt.ylabel('weight')
    # plt.show()
    # 아무리 길이가 늘어난들 저장된 데이터의 무게의 평균이 1033.33이 최대다
    # print(np.mean(train_target[indexes]))
    # 이런 문제를 해결하는 선형 회귀
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(train_input, train_target)
    print(lr.predict([[50]]))
    # 가중치와 절편 -> 모델 파라미터라고도 부름
    print(lr.coef_, lr.intercept_)
    # plt.scatter(train_input, train_target)
    # plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])
    # plt.scatter(50, 1241.8, marker='^')
    # plt.show()
    print(lr.score(train_input, train_target))
    print(lr.score(test_input, test_target))
    # 다항 회귀
    # 2차항을 붙이기 위한 column_stack()
    # train_input**2 : 넘파이 브로드캐스팅 -> train_input의 모든 원소를 제곱한다
    train_poly = np.column_stack((train_input**2, train_input))
    test_poly = np.column_stack((test_input**2, test_input))
    print(train_poly.shape, test_poly.shape)
    lr.fit(train_poly, train_target)
    print(lr.predict([[50**2, 50]]))
    # 다항도 선형 회귀다...
    # 길이의 제곱을 왕길이로 표현한다면
    # 무게(y) = 1.01 X 왕길이 - 21.6 X 길이 + 116.05로 쓸 수 있고
    # 즉 무게는 왕길이와 길이의 선형 관계로 표현할 수 있다
    # 이런 방정식을 다항식(polynomial)이라 부르고, 다항 회귀(polynomial regression)
    print(lr.coef_, lr.intercept_)

    point = np.arange(15, 50)
    plt.scatter(train_input, train_target)
    plt.plot(point, 1.01*point**2-21.6*point+116.05)
    plt.scatter(50, 1574, marker='^')
    # plt.show()

    print(lr.score(train_poly, train_target))
    print(lr.score(test_poly, test_target))
