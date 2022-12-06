class UnsupervisedLearning():
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
        fruits_2d = fruits.reshape(-1, 100*100)
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=3, random_state=42)
        km.fit(fruits_2d)
        # print(km.labels_)
        # print(np.unique(km.labels_, return_counts=True))

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

        # draw_fruits(fruits[km.labels_ == 2])
        # draw_fruits(fruits[km.labels_ == 1])
        # draw_fruits(fruits[km.labels_ == 0])
        # draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)
        print(km.transform(fruits_2d[100:101]))
        print(km.predict(fruits_2d[100:101]))
        draw_fruits(fruits[100:101])
        print(km.n_iter_)
