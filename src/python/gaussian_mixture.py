import matplotlib.pyplot as plt
import numpy as np

class GaussianMixture(object):

    def __init__(self, n_component):
        # Number of gaussian distribution 
        self.n_component = n_component

    # EMアルゴリズムを用いた最尤推定
    def fit(self, X, iter_max=100):
        '''
        prams:X : input_vectors(all_points * dim)
        '''
        # Dimension of data
        self.ndim = np.size(X, 1)
        print("DIM is {}".format(self.ndim))
        # 混合係数の初期化
        self.weights = np.ones(self.n_component) / self.n_component
        # 平均の初期化(入力の最大値と最小値の間のランダム値でdim*componentの行列をつくる)
        self.means = np.random.uniform(X.min(), X.max(), (self.ndim, self.n_component))
        # 共分散行列の初期化
        self.covs = np.repeat(10 * np.eye(self.ndim), self.n_component).reshape(self.ndim, self.ndim, self.n_component)

        # EステップとMステップを繰り返す
        for i in range(iter_max):
            params = np.hstack((self.weights.ravel(), self.means.ravel(), self.covs.ravel()))
            # Eステップ、負担率を計算
            resps = self.expectation(X)
            # Mステップ、パラメータを更新
            self.maximization(X, resps)
            # パラメータが収束したかを確認
            if np.allclose(params, np.hstack((self.weights.ravel(), self.means.ravel(), self.covs.ravel()))):
                break
            else:
                print("parameters may not have converged")

    # ガウス関数
    def gauss(self, X):
        precisions = np.linalg.inv(self.covs.T).T
        diffs = X[:, :, None] - self.means
        assert diffs.shape == (len(X), self.ndim, self.n_component)
        exponents = np.sum(np.einsum('nik,ijk->njk', diffs, precisions) * diffs, axis=1)
        assert exponents.shape == (len(X), self.n_component)
        return np.exp(-0.5 * exponents) / np.sqrt(np.linalg.det(self.covs.T).T * (2 * np.pi) ** self.ndim)

    #Eステップ
    def expectation(self, X):
        resps = self.weights * self.gauss(X)
        resps /= resps.sum(axis=-1, keepdims=True)
        return resps
    
    #Mステップ
    def maximization(self, X, resps):
        Nk = np.sum(resps, axis=0)
        self.weights = Nk / len(X)
        self.means = X.T.dot(resps) / Nk
        diffs = X[:, :, None] - self.means
        self.covs = np.einsum('nik,njk->ijk', diffs, diffs * np.expand_dims(resps, 1)) / Nk

    # 確率分布p(x)を計算
    def predict_proba(self, X):
        gauss = self.weights * self.gauss(X)
        return np.sum(gauss, axis=-1)

    # クラスタリング
    def classify(self, X):
        joint_prob = self.weights * self.gauss(X)
        return np.argmax(joint_prob, axis=1)


def create_toy_data():
    x1 = np.random.normal(size=(100, 2))
    x1 += np.array([-1, -1])
    x2 = np.random.normal(size=(100, 2))
    x2 += np.array([1, -1])
    x3 = np.random.normal(size=(100, 2))
    x3 += np.array([0, 2])
    return np.vstack((x1, x2, x3))


def main():
    X = create_toy_data()

    model = GaussianMixture(4)
    model.fit(X, iter_max=10000)
    labels = model.classify(X)

    x_test, y_test = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    X_test = np.array([x_test, y_test]).reshape(2, -1).transpose()
    probs = model.predict_proba(X_test)
    Probs = probs.reshape(100, 100)
    colors = ["red", "blue", "green", "orange"]
    plt.scatter(X[:, 0], X[:, 1], c=[colors[int(label)] for label in labels])
    # plt.contour(x_test, y_test, Probs)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()


if __name__ == '__main__':
    main()