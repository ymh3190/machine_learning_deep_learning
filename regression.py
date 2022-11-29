from feature import *


class Regression():
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

    def regress_linear_multiple() -> None:
        """여러 개의 특성을 사용한 선형 회귀를 다중 회귀라 한다
        - 하나의 특성을 사용한 선형 회귀 모델이 직선이라면 특성이 2개면 평면이 된다
        - 농어의 길이 뿐만 아니라 농어의 높이까지 특성에 포함
        """
        import pandas as pd
        # 인터넷에서 데이터를 바로 다운받아 사용
        # csv는 콤마로 나누어져 있는 텍스트 파일
        df = pd.read_csv('https://bit.ly/perch_csv_data')
        perch_full = df.to_numpy()
        # print(perch_full)
        # feature.py에 perch_weight가 정의됐으므로 생략
        # import numpy as np
        # perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
        #                      115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
        #                      150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
        #                      218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
        #                      556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
        #                      850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
        #                      1000.0])
        from sklearn.model_selection import train_test_split
        train_input, test_input, train_target, test_target = train_test_split(
            perch_full, perch_weight, random_state=42)
        # 특성을 만들거나 전처리하기
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures()
        # target이 필요없다
        poly.fit([[2, 3]])
        # fit을 해야 transform이 가능
        # 2개의 특성을 가진 샘플 [2, 3]이 6개의 특성을 가진 샘플 [1, 2, 3, 4, 6, 9]로 변환
        # 1이 추가되는 이유는 절편X1 때문
        # include_bias=False로 해결
        # print(poly.transform([[2, 3]]))
        poly = PolynomialFeatures(include_bias=False)
        poly.fit([[2, 3]])
        # print(poly.transform([[2, 3]]))
        poly = PolynomialFeatures(include_bias=False)
        poly.fit(train_input)
        train_poly = poly.transform(train_input)
        # print(train_poly.shape)
        # 특성이 어떻게 만들어졌는지 확인하는 방법 제공
        # 'x0' 'x1' 'x2' 'x0^2' 'x0 x1' 'x0 x2' 'x1^2' 'x1 x2' 'x2^2'
        # print(poly.get_feature_names_out())
        # 항상 훈련 세트를 기준으로 테스트 세트를 변환(fit)
        test_poly = poly.transform(test_input)
        # print(test_poly)

        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(train_poly, train_target)
        # print(lr.score(train_poly, train_target))
        # print(lr.score(test_poly, test_target))
        # 5제곱까지 특성을 만들어 추가
        poly = PolynomialFeatures(degree=5, include_bias=False)
        poly.fit(train_input)
        train_poly = poly.transform(train_input)
        test_poly = poly.transform(test_input)
        # print(train_poly.shape)

        lr.fit(train_poly, train_target)
        # print(lr.score(train_poly, train_target))
        # 테스트 점수는 음수..
        # 특성을 많게 한다고 능사가 아니다
        # print(lr.score(test_poly, test_target))

        # regularization(규제) : 머신 러닝 모델이 훈련 세트를 너무 과도하게 학습하지 못하도록 훼방하는 것
        # 선형 회귀 모델의 경우 특성에 곱해지는 계수(가중치)의 크기를 작게 만드는 일
        # 모델이 훈련 세트에 과대적합되지 않도록 하는 것
        # 특성 스케일
        from sklearn.preprocessing import StandardScaler
        ss = StandardScaler()
        ss.fit(train_poly)
        train_scaled = ss.transform(train_poly)
        test_scaled = ss.transform(test_poly)

        # ridge(릿지 회귀)
        from sklearn.linear_model import Ridge
        ridge = Ridge()
        ridge.fit(train_scaled, train_target)
        # print(ridge.score(train_scaled, train_target))
        # print(ridge.score(test_scaled, test_target))

        # 모델이 학습할 수 없고 사람이 알려줘야 하는 파라미터를 하이퍼파라미터라 한다
        # alpha가 크면 규제 강도가 쎄짐
        # 적절한 alpha값 찾기
        import matplotlib.pyplot as plt
        train_score = []
        test_score = []
        alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
        for alpha in alpha_list:
            ridge = Ridge(alpha=alpha)
            ridge.fit(train_scaled, train_target)
            train_score.append(ridge.score(train_scaled, train_target))
            test_score.append(ridge.score(test_scaled, test_target))

        # plt.plot(np.log10(alpha_list), train_score)
        # plt.plot(np.log10(alpha_list), test_score)
        plt.xlabel('alpha')
        plt.ylabel('R^2')
        # plt.show()
        ridge = Ridge(alpha=0.1)
        ridge.fit(train_scaled, train_target)
        # print(ridge.score(train_scaled, train_target))
        # print(ridge.score(test_scaled, test_target))

        # lasso(라쏘 회귀)
        # 훈련 세트 억제
        # 가중치를 0으로 만들 수 있음
        from sklearn.linear_model import Lasso
        lasso = Lasso()
        lasso.fit(train_scaled, train_target)
        # print(lasso.score(train_scaled, train_target))
        train_score = []
        test_score = []
        alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
        for alpha in alpha_list:
            lasso = Lasso(alpha=alpha, max_iter=100000000)
            lasso.fit(train_scaled, train_target)
            train_score.append(lasso.score(train_scaled, train_target))
            test_score.append(lasso.score(test_scaled, test_target))

        plt.plot(np.log10(alpha_list), train_score)
        plt.plot(np.log10(alpha_list), test_score)
        # plt.show()
        lasso = Lasso(alpha=10)
        lasso.fit(train_scaled, train_target)
        print(lasso.score(train_scaled, train_target))
        print(lasso.score(test_scaled, test_target))

        # 0이 된 가중치
        print(np.sum(lasso.coef_ == 0))
