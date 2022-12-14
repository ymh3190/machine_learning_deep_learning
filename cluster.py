class Cluster():
    def cluster() -> None:
        """ 비지도학습이지만 어떤 사진인지 알고있었다. """
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
        # pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
        # banana_mean = np.mean(banana, axis=0).reshape(100, 100)
        # fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        # axs[0].imshow(apple_mean, cmap='gray_r')
        # axs[1].imshow(pineapple_mean, cmap='gray_r')
        # axs[2].imshow(banana_mean, cmap='gray_r')
        # plt.show()
        abs_diff = np.abs(fruits-apple_mean)
        abs_mean = np.mean(abs_diff, axis=(1, 2))
        # print(abs_mean.shape)
        apple_index = np.argsort(abs_mean)[:100]
        fig, axs = plt.subplots(10, 10, figsize=(10, 10))
        for i in range(10):
            for j in range(10):
                axs[i, j].imshow(fruits[apple_index[i*10+j]], cmap='gray_r')
                axs[i, j].axis('off')
        plt.show()

    def cluster_center() -> None:
        """진짜 비지도 학습에서는 사진에 어떤 과일이 들어 있는지 알지 못한다.

        k-means(k-평균) 군집 알고리즘

        cluster center 혹은 centroid라고 부름
        """
        import os
        FRUITS_300_npy = 'fruits_300.npy'
        if FRUITS_300_npy in os.listdir(os.getcwd()):
            pass
        else:
            import wget
            wget.download('https://bit.ly/fruits_300_data', out=FRUITS_300_npy)

        import numpy as np
        fruits = np.load(FRUITS_300_npy)
        print(fruits)
        print(fruits.shape)

        fruits_2d = fruits.reshape(-1, 100*100)
        print(fruits_2d)
        print(fruits_2d.shape)

        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=3, random_state=42)
        km.fit(fruits_2d)
        print(km.labels_)
        print(np.unique(km.labels_, return_counts=True))

        draw_fruits(fruits[km.labels_ == 2])
        draw_fruits(fruits[km.labels_ == 1])
        draw_fruits(fruits[km.labels_ == 0])
        draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)

        print(km.transform(fruits_2d[100:101]))
        print(km.predict(fruits_2d[100:101]))
        draw_fruits(fruits[100:101])
        print(km.n_iter_)

        # 최적의 k 찾기
        inertia = []
        for k in range(2, 7):
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(fruits_2d)
            inertia.append(km.inertia_)

        import matplotlib.pyplot as plt
        plt.plot(range(2, 7), inertia)
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.show()

    def dimension_reduction() -> None:
        """차원축소 : 3차원을 2차원으로 줄이는게 아니라
        1차원 배열에서 열의 개수를 줄이는 것을 의미한다
        데이터를 가장 잘 나타내는 일부 특성을 선택하여
        데이터 크기를 줄이고 지도 학습 모델의 성능을 향상시키는 방법
        PCA(principal component analysis) : 주성분 분석
        """
        import os
        FRUITS_300_npy = 'fruits_300.npy'
        if FRUITS_300_npy in os.listdir(os.getcwd()):
            pass
        else:
            import wget
            wget.download('https://bit.ly/fruits_300_data', out=FRUITS_300_npy)
        import numpy as np
        fruits = np.load(FRUITS_300_npy)
        fruits_2d = fruits.reshape(-1, 100*100)
        from sklearn.decomposition import PCA
        """ 주성분 개수 지정 """
        pca = PCA(n_components=50)
        pca.fit(fruits_2d)
        # print(pca.components_.shape)
        # draw_fruits(pca.components_.reshape(-1, 100, 100))
        # print(fruits_2d.shape)
        fruits_pca = pca.transform(fruits_2d)
        # print(fruits_pca.shape)

        fruits_inverse = pca.inverse_transform(fruits_pca)
        # print(fruits_inverse.shape)

        fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
        for start in [0, 100, 200]:
            # draw_fruits(fruits_reconstruct[start:start+100])
            continue
        # print(np.sum(pca.explained_variance_ratio_))
        import matplotlib.pyplot as plt
        # plt.plot(pca.explained_variance_ratio_)
        # plt.show()

        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        target = np.array([0]*100+[1]*100+[1]*100)
        from sklearn.model_selection import cross_validate
        scores = cross_validate(lr, fruits_2d, target)
        print(np.mean(scores['test_score']))
        print(np.mean(scores['fit_time']))
        scores = cross_validate(lr, fruits_pca, target)
        print(np.mean(scores['test_score']))
        print(np.mean(scores['fit_time']))

        """ 설명된 분산 비율 입력 """
        pca = PCA(n_components=0.5)
        pca.fit(fruits_2d)
        print(pca.n_components_)
        fruits_pca = pca.transform(fruits_2d)
        print(fruits_pca.shape)
        scores = cross_validate(lr, fruits_pca, target)
        print(np.mean(scores['test_score']))
        print(np.mean(scores['fit_time']))

        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=3, random_state=42)
        km.fit(fruits_pca)
        # print(np.unique(km.labels_,return_counts=True))
        for label in range(3):
            draw_fruits(fruits[km.labels_ == label])

        for label in range(3):
            data = fruits_pca[km.labels_ == label]
            plt.scatter(data[:, 0], data[:, 1])
        plt.legend(['apple', 'banana', 'pineapple'])
        plt.show()


def draw_fruits(arr, ratio=1):
    import matplotlib.pyplot as plt
    import numpy as np
    n = len(arr)
    rows = int(np.ceil(n/10))
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, figsize=(
        cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:
                axs[i, j].imshow(arr[i*10+j], cmap='gray_r')
            axs[i, j].axis('off')

    plt.show()
