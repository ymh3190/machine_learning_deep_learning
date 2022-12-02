from feature import *


class Classification():
    def classify() -> None:
        """첫 번째 머신 러닝 프로그램 : k-최근접 이웃(k-Nearest Neighbors) 알고리즘
        - 어떤 데이터에 대한 답을 구할 때 주위의 다른 데이터를 보고 다수를 차지하는 것을 정답으로 합니다.
        - 단점 : 새로운 데이터에 대해 예측할 때는 가장 가까운 직선거리에 어떤 데이터가 있는지를 살핀다
            이러한 특징으로 데이터가 아주 많은 경우 사용하기 어려움
        - 사례 기반 학습
        """
        length = bream_length+smelt_length
        weight = bream_weight+smelt_weight
        fish_data = [[l, w] for l, w in zip(length, weight)]
        fish_target = [1]*35 + [0]*14
        from sklearn.neighbors import KNeighborsClassifier
        kn = KNeighborsClassifier()
        """ training """
        kn.fit(fish_data, fish_target)
        """ 평가 """
        # print(kn.score(fish_data, fish_target))
        """ 예측 """
        # print(kn.predict([[30, 600]]))
        """ 매개변수 _fit_X = fish_data """
        # print(kn._fit_X)
        """ 매개변수 _y = fish_target """
        # print(kn._y)
        """ 기본 참고 데이터 개수 5개 """
        kn49 = KNeighborsClassifier(n_neighbors=49)
        kn49.fit(fish_data, fish_target)
        # print(kn49.score(fish_data, fish_target))
        """ 슬라이싱 """
        # print(fish_data[4])
        # print(fish_data[:5])
        # print(fish_data[44:])
        train_input = fish_data[:35]
        train_target = fish_target[:35]
        test_input = fish_data[35:]
        test_target = fish_target[35:]
        kn = kn.fit(train_input, train_target)
        """ 샘플링 편향(sampling bias)으로 인해 평가결과 0점 """
        # print(kn.score(test_input, test_target))
        """ 샘플링 편향 문제를 해결하기 위한 numpy(넘파이)
        리스트를 넘파이 배열로 변경 """
        import numpy as np
        input_arr = np.array(fish_data)
        target_arr = np.array(fish_target)
        # print(input_arr)
        """ 배열의 크기를 알려주는 shape 속성(샘플 수, feature 수) """
        # print(input_arr.shape)
        """ 42는 넘파이 난수 생성에서 흔히 쓰임 """
        np.random.seed(42)
        """ 0~48까지 1씩 증가하는 리스트를 만듦 """
        index = np.arange(49)
        """ 주어진 배열을 무작위로 섞는다 """
        np.random.shuffle(index)
        # print(index)
        """ array indexing(배열 인덱싱) : 여러 개의 원소 선택 """
        # print(input_arr[[1, 3]])
        train_input = input_arr[index[:35]]
        train_target = target_arr[index[:35]]
        # print(input_arr[13], train_input[0])
        test_input = input_arr[index[35:]]
        test_target = target_arr[index[35:]]
        import matplotlib.pyplot as plt
        """ 넘파이 2차원 배열은 행과 열 인덱스를 콤마로 나누어 지정할 수 있다 """
        # plt.scatter(train_input[:, 0], train_input[:, 1])
        # plt.scatter(test_input[:, 0], test_input[:, 1])
        plt.xlabel('length')
        plt.ylabel('weight')
        # plt.show()
        # print(kn.score(test_input, test_target))
        kn = kn.fit(train_input, train_target)
        """ numpy 배열 반환, 파이썬 리스트 아님 """
        # print(kn.predict(test_input))
        # print(test_target)
        """ 파이썬 리스트 순회없이 넘파이를 이용한 데이터 구성 """
        np.column_stack(([1, 2, 3, ], [4, 5, 6]))
        fish_data = np.column_stack((fish_length, fish_weight))
        # print(fish_data[:5])
        # np.ones()와 np.zeros()
        # print(np.ones(5))
        # concatenate()
        fish_target = np.concatenate((np.ones(35), np.zeros(14)))
        # print(fish_target)
        """ 사이킷런으로 훈련 세트와 테스트 세트 나누기 """
        from sklearn.model_selection import train_test_split
        """ 기본적으로 25%를 테스트 세트로 분리 """
        train_input, test_input, train_target, test_target = train_test_split(
            fish_data, fish_target, random_state=42)
        # print(train_input.shape, test_input.shape)
        # print(train_target.shape, test_target.shape)
        """ 개수가 적을 경우 비율이 맞지 않을 수 있다 """
        # print(test_target)
        """ stratify 매개변수로 조정 가능 """
        train_input, test_input, train_target, test_target = train_test_split(
            fish_data, fish_target, stratify=fish_target, random_state=42)

        print(test_target)
        kn.fit(train_input, train_target)
        print(kn.score(test_input, test_target))
        print(kn.predict([[25, 150]]))
        """ 1이 아닌 0으로 예측 왜? """
        # plt.scatter(train_input[:, 0], train_input[:, 1])
        # plt.scatter(25, 150, marker='^')
        # plt.show()
        """ 주어진 샘플에서 가장 가까운 이웃을 찾는 kneighbors(), 이웃까지의 거리와 이웃 샘플 인덱스 반환 """
        """ n_neighbors의 기본값은 5이므로 5개 이웃을 반환 """
        distances, indexes = kn.kneighbors([[25, 150]])
        # plt.scatter(train_input[:, 0], train_input[:, 1])
        # plt.scatter(25, 150, marker='^')
        # plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
        # plt.show()
        # print(train_input[indexes])
        # print(train_target[indexes])
        # print(distances)
        """ x축 범위 지정 """
        # plt.xlim((0, 1000))
        # plt.show()
        """ 데이터 전처리 : 데이터를 표현하는 기준을 맞춤(인치, 센티 등)
        표준점수(standard score) : 가장 널리 사용하는 전처리 방법 중 하나
        mean() : 평균 계산
        std() : 표준편차 계산
        axis=0 : 열 계산, axis=1 행 계산 """
        mean = np.mean(train_input, axis=0)
        std = np.std(train_input, axis=0)
        # print(mean, std)
        """ 브로드캐스팅 : 넘파이 배열 사이에서 일어나는 연산 """
        train_scaled = (train_input - mean)/std
        # print(train_scaled)
        new = ([25, 150]-mean)/std
        plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
        plt.scatter(new[0], new[1], marker='^')
        # plt.show()
        kn.fit(train_scaled, train_target)
        test_scaled = (test_input-mean)/std
        # print(kn.score(test_scaled, test_target))
        # print(kn.predict([new]))
        distances, indexes = kn.kneighbors([new])
        plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
        plt.scatter(new[0], new[1], marker='^')
        plt.scatter(train_scaled[indexes, 0],
                    train_scaled[indexes, 1], marker='D')
        # plt.show()

    def classify_logistic_regression() -> None:
        """이름은 회귀지만 분류 모델
        - 선형 회귀와 동일하게 선형 방정식을 학습하지만 타겟을 분류(확률을 통해)하는데 사용
        """
        import pandas as pd
        """ 데이터프레임 """
        fish = pd.read_csv('https://bit.ly/fish_csv_data')
        """ 처음 5개 행 출력 """
        print(fish.head())
        """ 생선 종류 확인 """
        print(pd.unique(fish['Species']))

        """ 입력 데이터 선택 """
        fish_input = fish[['Weight', 'Length',
                           'Diagonal', 'Height', 'Width']].to_numpy()
        # print(fish_input[:5])
        fish_target = fish['Species'].to_numpy()
        from sklearn.model_selection import train_test_split
        train_input, test_input, train_target, test_target = train_test_split(
            fish_input, fish_target, random_state=42)
        from sklearn.preprocessing import StandardScaler
        ss = StandardScaler()
        ss.fit(train_input)
        train_scaled = ss.transform(train_input)
        test_scaled = ss.transform(test_input)
        from sklearn.neighbors import KNeighborsClassifier
        kn = KNeighborsClassifier(n_neighbors=3)
        kn.fit(train_scaled, train_target)
        # print(kn.score(train_scaled, train_target))
        # print(kn.score(test_scaled, test_target))
        """ 타겟 데이터에 2개 이상의 클래스가 포함된 문제 -> 다중 분류(multi-class classification) """
        """ 이진 분류와 모델을 만들고 훈련하는 방식은 동일 """
        # print(kn.classes_)
        # print(kn.predict(test_scaled[:5]))
        proba = kn.predict_proba(test_scaled[:5])
        import numpy as np
        # print(np.round(proba, decimals=4))
        distances, indexes = kn.kneighbors(test_scaled[3:4])
        """ 분류 근거가 부실한 단점(1/3과 2/3와 같은 기준) """
        # print(train_target[indexes])

        import matplotlib.pyplot as plt
        z = np.arange(-5, 5, 0.1)
        """ 시그모이드 함수(sigmoid function) 또는 로지스틱 함수(logistic function) """
        phi = 1/(1+(np.exp(-z)))
        # plt.plot(z, phi)
        # plt.xlabel('z')
        # plt.ylabel('phi')
        # plt.show()

        char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
        # print(char_arr[[True, False, True, False, False]])

        """ 넘파이 배열 불리언 인덱싱 """
        """ true false로 값을 전달하여 행을 선택할 수 잇다 """
        bream_smelt_indexes = (train_target == 'Bream') | (
            train_target == 'Smelt')
        train_bream_smelt = train_scaled[bream_smelt_indexes]
        target_bream_smelt = train_target[bream_smelt_indexes]
        # print(target_bream_smelt)

        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        lr.fit(train_bream_smelt, target_bream_smelt)
        # print(lr.predict(train_bream_smelt[:5]))
        # 첫 번째 열이 음성 클래스(0), 두 번째 열이 양성 클래스(1)
        # print(lr.predict_proba(train_bream_smelt[:5]))
        # print(lr.classes_)
        # 로지스틱 회귀가 학습한 계수
        # print(lr.coef_, lr.intercept_)
        decisions = lr.decision_function(train_bream_smelt[:5])
        # print(decisions)

        # 시그모이드 함수 파이썬 사이파이 라이브러리
        # scipy
        # predict_proba() 메서드 출력의 두 번째의 값과 동일
        from scipy.special import expit
        # print(expit(decisions))

        lr = LogisticRegression(C=20, max_iter=1000)
        lr.fit(train_scaled, train_target)
        # print(lr.score(train_scaled, train_target))
        # print(lr.score(test_scaled, test_target))

        # print(lr.predict(test_scaled[:5]))
        proba = lr.predict_proba(test_scaled[:5])
        # print(np.round(proba, decimals=3))

        # print(lr.coef_.shape, lr.intercept_.shape)

        decision = lr.decision_function(test_scaled[:5])
        # print(np.round(decision, decimals=2))

        # softmax()
        from scipy.special import softmax
        proba = softmax(decision, axis=1)
        # print(np.round(proba, decimals=3))

    def classify_stochastic_gradient_descent() -> None:
        """앞서 훈련한 모델을 버리지 않고 새로운 데이터에 대해서만 조금씩 더 훈련하는 방법

        확률적 경사 하강법(Stochastic Gradient Descent)
        --------
        훈련 세트에서 랜덤하게 하나의 샘플을 골라 모델을 학습

        미니배치 경사 하강법(minibatch gradient descent)
        --------
        훈련 세트에서 랜덤하게 여러 개의 샘플을 골라 모델을 학습

        배치 경사 하강법(batch gradient descent)
        --------
        훈련 세트 전부를 골라 모델을 학습

        - SGDRegression: 확률적 경사 하강법을 사용한 회귀 알고리즘
        """
        import pandas as pd
        fish = pd.read_csv('https://bit.ly/fish_csv_data')
        fish_input = fish[['Weight', 'Length',
                           'Diagonal', 'Height', 'Width']].to_numpy()
        fish_target = fish['Species'].to_numpy()
        from sklearn.model_selection import train_test_split
        train_input, test_input, train_target, test_target = train_test_split(
            fish_input, fish_target, random_state=42)
        from sklearn.preprocessing import StandardScaler
        ss = StandardScaler()
        ss.fit(train_input)
        train_scaled = ss.transform(train_input)
        test_scaled = ss.transform(test_input)
        from sklearn.linear_model import SGDClassifier
        sc = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
        sc.fit(train_scaled, train_target)
        sc.partial_fit(train_scaled, train_target)
        import numpy as np
        sc = SGDClassifier(loss='log_loss', random_state=42)
        train_score = []
        test_score = []
        classes = np.unique(train_target)
        for _ in range(300):
            sc.partial_fit(train_scaled, train_target, classes=classes)
            train_score.append(sc.score(train_scaled, train_target))
            test_score.append(sc.score(test_scaled, test_target))
        import matplotlib.pyplot as plt
        plt.plot(train_score)
        plt.plot(test_score)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        sc = SGDClassifier(loss='log_loss', max_iter=100,
                           tol=None, random_state=42)
        sc.fit(train_scaled, train_target)
        # print(sc.score(train_scaled, train_target))
        # print(sc.score(test_scaled, test_target))
        sc = SGDClassifier(loss='hinge', max_iter=100,
                           tol=None, random_state=42)
        sc.fit(train_scaled, train_target)
        # print(sc.score(train_scaled, train_target))
        # print(sc.score(test_scaled, test_target))

    def classify_decision_tree() -> None:
        """결정 트리(Decision Tree) 모델은 이유를 설명하기 쉽다.

        결정 트리 모델은 부모와 자식 노드의 불순도 차이가 가능한 크도록 성장시킨다.

        - (지니) gini: 불순도
        - (정보 이득) information gain: 부모와 자식 노드의 불순도 차이
        """
        import pandas as pd
        wine = pd.read_csv('https://bit.ly/wine_csv_data')
        # print(wine.head())
        # print(wine.info())
        # print(wine.describe())
        data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
        target = wine['class'].to_numpy()
        from sklearn.model_selection import train_test_split
        train_input, test_input, train_target, test_target = train_test_split(
            data, target, test_size=0.2, random_state=42)
        from sklearn.preprocessing import StandardScaler
        ss = StandardScaler()
        ss.fit(train_input)
        train_scaled = ss.transform(train_input)
        test_scaled = ss.transform(test_input)
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        lr.fit(train_scaled, train_target)
        from sklearn.tree import DecisionTreeClassifier
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(train_scaled, train_target)
        # print(dt.score(train_scaled, train_target))
        # print(dt.score(test_scaled, test_target))
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree
        # plt.figure(figsize=(10, 7))
        # plot_tree(dt, max_depth=1, filled=True,
        #           feature_names=['alcohol', 'sugar', 'pH'])
        # plt.show()
        """ 일반화(가지치기): 최대 깊이 지정 """
        dt = DecisionTreeClassifier(max_depth=3, random_state=42)
        # dt.fit(train_scaled, train_target)
        # print(dt.score(train_scaled, train_target))
        # print(dt.score(test_input, test_target))
        # plt.figure(figsize=(20, 15))
        # plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
        # plt.show()
        """ gini는 클래스별 비율로 계산했기에 전처리 과정이 필요없다 """
        dt.fit(train_input, train_target)
        # print(dt.score(train_input, train_target))
        # print(dt.score(test_input, test_target))
        plt.figure(figsize=(20, 15))
        plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
        plt.show()
        """ 어떤 특성이 가장 유용한지 나타내는 특성 중요도 """
        # print(dt.feature_importances_)

    def classify_cross_validation() -> None:
        """
        모델 파라미터
        - 머신 러닝 모델이 학습하는 파라미터
        ------------------------------------

        하이퍼 파라미터
        - 모델이 학습할 수 없어서 사용자가 지정해야만 하는 파라미터
        """
        import pandas as pd
        wine = pd.read_csv('https://bit.ly/wine_csv_data')
        data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
        target = wine['class'].to_numpy()
        from sklearn.model_selection import train_test_split
        train_input, test_input, train_target, test_target = train_test_split(
            data, target, test_size=0.2, random_state=42)
        sub_input, val_input, sub_target, val_target = train_test_split(
            train_input, train_target, test_size=0.2, random_state=42)
        # print(sub_input.shape, val_input.shape)
        from sklearn.tree import DecisionTreeClassifier
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(sub_input, sub_target)
        # print(dt.score(sub_input, sub_target))
        # print(dt.score(val_input, val_target))
        """ k-폴드 교차 검증(k-겹 교차 검증) """
        from sklearn.model_selection import cross_validate
        scores = cross_validate(dt, train_input, train_target)
        # print(scores)
        import numpy as np
        # print(np.mean(scores['test_score']))
        """ Splitter: 사이킷런 분할기는 교차 검증에서 훈련 세트를 섞을 때 사용 """
        from sklearn.model_selection import StratifiedKFold
        scores = cross_validate(
            dt, train_input, train_target, cv=StratifiedKFold())
        # print(np.mean(scores['test_score']))
        """ 훈련 세트를 섞은 후 10-폴드 교차 검증 """
        splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_validate(dt, train_input, train_target, cv=splitter)
        # print(np.mean(scores['test_score']))
        """ grid search(그리드 서치) """
        from sklearn.model_selection import GridSearchCV
        params = {'min_impurity_decrease': [
            0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
        gs = GridSearchCV(DecisionTreeClassifier(
            random_state=42), params, n_jobs=-1)
        gs.fit(train_input, train_target)
        """ 점수가 가장 높은 모델 """
        dt = gs.best_estimator_
        # print(dt.score(train_input, train_target))
        """ 그리드 서치로 찾은 최적의 매개변수 """
        # print(gs.best_params_)
        """ 각 매개변수에서 수행한 교차 점증의 평균 점수 """
        # print(gs.cv_results_['mean_test_score'])
        params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001), 'max_depth': range(
            5, 20, 1), 'min_samples_split': range(2, 100, 10)}
        gs = GridSearchCV(DecisionTreeClassifier(
            random_state=42), params, n_jobs=-1)
        gs.fit(train_input, train_target)
        # print(gs.best_params_)
        """최상의 교차 검증 점수 """
        # print(np.max(gs.cv_results_['mean_test_score']))
        """랜덤 서치: 매개변수 값의 목록을 전달하는게 아니라  매개변수를 샘플링할 수 있는 확률 분포 객체를 전달"""
        from scipy.stats import uniform, randint
        rgen = randint(0, 10)
        # print(rgen.rvs(10))
        # print(np.unique(rgen.rvs(1000), return_counts=True))
        ugen = uniform(0, 1)
        # print(ugen.rvs(10))
        params = {'min_impurity_decrease': uniform(0.0001, 0.001), 'max_depth': range(
            20, 50), 'min_samples_split': randint(2, 25), 'min_samples_leaf': randint(1, 25)}
        from sklearn.model_selection import RandomizedSearchCV
        gs = RandomizedSearchCV(DecisionTreeClassifier(
            random_state=42), params, n_iter=100, n_jobs=-1, random_state=42)
        gs.fit(train_input, train_target)
        # print(gs.best_params_)
        # print(np.max(gs.cv_results_['mean_test_score']))
        dt = gs.best_estimator_
        # print(dt.score(test_input, test_target))``

    def classify_tree_ensemble() -> None:
        """정형 데이터(structured data)에 가장 뛰어난 성과를 내는 알고리즘"""
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        wine = pd.read_csv('https://bit.ly/wine_csv_data')
        data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
        target = wine['class'].to_numpy()
        train_input, test_input, train_target, test_target = train_test_split(
            data, target, test_size=0.2, random_state=42)
        from sklearn.model_selection import cross_validate
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_jobs=-1, random_state=42)
        scores = cross_validate(
            rf, train_input, train_target, return_train_score=True, n_jobs=-1)
        # print(np.mean(scores['train_score']), np.mean(scores['test_score']))
        # rf.fit(train_input, train_target)
        # print(rf.feature_importances_)
        rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
        rf.fit(train_input, train_target)
        # print(rf.oob_score_)
        """ 엑스트라 앙상블 """
        from sklearn.ensemble import ExtraTreesClassifier
        et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
        scores = cross_validate(
            et, train_input, train_target, return_train_score=True, n_jobs=-1)
        # print(np.mean(scores['train_score']), np.mean(scores['test_score']))
        et.fit(train_input, train_target)
        # print(et.feature_importances_)
        # print(et.feature_importances_)

    def classify_gradient_boosting() -> None:
        """그레디언트 부스팅 : 경사 하강법을 사용하여 트리의 앙상블에 추가"""
        from sklearn.ensemble import GradientBoostingClassifier
        gb = GradientBoostingClassifier(random_state=42)
        from sklearn.model_selection import cross_validate
        import pandas as pd
        wine = pd.read_csv('https://bit.ly/wine_csv_data')
        data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
        target = wine['class'].to_numpy()
        from sklearn.model_selection import train_test_split
        train_input, test_input, train_target, test_target = train_test_split(
            data, target, test_size=0.2, random_state=42)
        scores = cross_validate(
            gb, train_input, train_target, return_train_score=True, n_jobs=-1)
        import numpy as np
        # print(np.mean(scores['train_score']), np.mean(scores['test_score']))
        gb = GradientBoostingClassifier(
            n_estimators=500, learning_rate=0.2, random_state=42)
        scores = cross_validate(
            gb, train_input, train_target, return_train_score=True, n_jobs=-1)
        # print(np.mean(scores['train_score']), np.mean(scores['test_score']))
        gb.fit(train_input, train_target)
        # print(gb.feature_importances_)

    def classify_histogram_based_gradient_boosting() -> None:
        """히스토그램 기반 그레디언트 부스팅 : 입력 특성을 256개 구간으로 나눔"""
        from sklearn.ensemble import HistGradientBoostingClassifier
        hgb = HistGradientBoostingClassifier(random_state=42)
        from sklearn.model_selection import cross_validate, train_test_split
        import pandas as pd
        import numpy as np
        wine = pd.read_csv('https://bit.ly/wine_csv_data')
        data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
        target = wine['class'].to_numpy()
        train_input, test_input, train_target, test_target = train_test_split(
            data, target, test_size=0.2, random_state=42)
        scores = cross_validate(
            hgb, train_input, train_target, return_train_score=True)
        # print(np.mean(scores['train_score']), np.mean(scores['test_score']))
        from sklearn.inspection import permutation_importance
        hgb.fit(train_input, train_target)
        result = permutation_importance(
            hgb, train_input, train_target, n_repeats=10, random_state=42, n_jobs=-1)
        # print(result.importances_mean)
        result = permutation_importance(
            hgb, test_input, test_target, n_repeats=10, random_state=42, n_jobs=-1)
        # print(result.importances_mean)
        # print(hgb.score(test_input, test_target))

        """ XGBoost Library """
        from xgboost import XGBClassifier
        xgb = XGBClassifier(tree_method='hist', random_state=42)
        scores = cross_validate(
            xgb, train_input, train_target, return_train_score=True)
        # print(np.mean(scores['train_score']), np.mean(scores['test_score']))

        """ LightGBM Library by Microsoft  """
        from lightgbm import LGBMClassifier
        lgb = LGBMClassifier(random_state=42)
        scores = cross_validate(
            lgb, train_input, train_target, return_train_score=True, n_jobs=-1)
        # print(np.mean(scores['train_score']), np.mean(scores['test_score']))
