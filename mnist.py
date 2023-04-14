import numpy as np
import torchvision

# sigmoid & 一阶导数
sigmoid = lambda z: 1 / (1 + np.exp(-z))
derivative_sigmoid = lambda z: sigmoid(z) * (1 - sigmoid(z))


class NeuralNetwork(object):
    def __init__(self, l0, l1, l2, l3, batch_size=10):
        """
        初始化神经网络
        :param l0: 输入层节点数
        :param l1: 隐含层 l1 节点数
        :param l2: 隐含层 l2 节点数
        :param l3: 输出层节点数量
        :param batch_size: 单次训练批次数
        """
        self.lr = 0.5  # 学习率
        self.batch_size = batch_size
        # 各层权重与偏置量
        self.w1 = np.random.randn(l0, l1) * 0.01
        self.b1 = np.random.randn(l1) * 0.01
        self.w2 = np.random.randn(l1, l2) * 0.01
        self.b2 = np.random.randn(l2) * 0.01
        self.w3 = np.random.randn(l2, l3) * 0.01
        self.b3 = np.random.randn(l3) * 0.01

    def forward(self, x):
        """
        向前传播推导结果
        :param x: 输入的 [784] 向量矩阵
        :return: 输出各层的净输入和激活值
        """
        z1 = np.dot(x, self.w1) + self.b1
        o1 = sigmoid(z1)

        z2 = np.dot(o1, self.w2) + self.b2
        o2 = sigmoid(z2)

        z3 = np.dot(o2, self.w3) + self.b3
        o3 = sigmoid(z3)
        return z1, o1, z2, o2, z3, o3

    def backward(self, x, z1, o1, z2, o2, err3):
        """
        反向传播更新权重
        """

        dot_w3 = np.dot(o2.T, err3) / self.batch_size
        dot_b3 = np.sum(err3, axis=0) / self.batch_size

        err2 = np.dot(err3, self.w3.T) * derivative_sigmoid(z2)
        dot_w2 = np.dot(o1.T, err2) / self.batch_size
        dot_b2 = np.sum(err2, axis=0) / self.batch_size

        err1 = np.dot(err2, self.w2.T) * derivative_sigmoid(z1)
        dot_w1 = np.dot(x.T, err1) / self.batch_size
        dot_b1 = np.sum(err1, axis=0) / self.batch_size

        self.w3 -= self.lr * dot_w3
        self.b3 -= self.lr * dot_b3
        self.w2 -= self.lr * dot_w2
        self.b2 -= self.lr * dot_b2
        self.w1 -= self.lr * dot_w1
        self.b1 -= self.lr * dot_b1


def train(nn, data, targets):
    for cou in range(10):
        for i in range(0, 60000, nn.batch_size):
            x = data[i:i + nn.batch_size]
            y = targets[i:i + nn.batch_size]
            z1, o1, z2, o2, z3, o3 = nn.forward(x)
            err3 = (o3 - y) * derivative_sigmoid(z3)
            loss = np.sum((o3 - y) * (o3 - y)) / nn.batch_size
            print("cou:" + str(cou) + ", err:" + str(loss))
            nn.backward(x, z1, o1, z2, o2, err3)


def test(nn, data, targets):
    _, _, _, _, _, o3 = nn.forward(data)
    result = np.argmax(o3, axis=1)
    precision = np.sum(result == targets) / 10000
    print("Precision:", precision)


def target_matrix(targets):
    """
    数字标签转换
    :param targets: 对于的数字标签矩阵
    :return:
    """
    num = len(targets)
    result = np.zeros((num, 10))
    for i in range(num):
        result[i][targets[i]] = 1
    return result


# 训练数据
def load_train_data():
    train_data = torchvision.datasets.MNIST(root='data/', train=True, download=True)
    # Numpy 矩阵转换
    train_data.data = train_data.data.numpy()  # [60000,28,28]
    train_data.targets = train_data.targets.numpy()  # [60000]
    # 输入向量处理，将二维数据平铺
    train_data.data = train_data.data.reshape(60000, 28 * 28) / 255.  # (60000, 784)
    # 标签转换
    train_data.targets = target_matrix(train_data.targets)  # (60000, 10)
    return train_data


# 测试数据
def load_test_data():
    test_data = torchvision.datasets.MNIST(root='data/', train=False)
    test_data.data = test_data.data.numpy()  # [10000,28,28]
    test_data.targets = test_data.targets.numpy()  # [10000]
    test_data.data = test_data.data.reshape(10000, 28 * 28) / 255.  # (10000, 784)
    return test_data


def demo():
    nn = NeuralNetwork(784, 200, 30, 10)
    train_data = load_train_data()
    train(nn, train_data.data, train_data.targets)
    test_data = load_test_data()
    test(nn, test_data.data, test_data.targets)


demo()
