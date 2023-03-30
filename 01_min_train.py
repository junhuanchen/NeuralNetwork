import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))


def grad(x):
    return x*(1-x)


class NeuralNetwork:
    """
    三层全连接前馈神经网络
    """

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate, active_function=sigmoid, gradient=grad, lambda_=0.1):
        """

        :param inputnodes: 输入层结点数
        :param hiddennodes: 隐藏层节点数
        :param outputnodes: 输出层节点数
        :param learningrate: 学习率
        :param active_function: 激活函数
        :param gradient: 激活函数的导数
        :param lambda_: L2正则化系数
        """
        self.inputnodes = inputnodes
        self.hiddennodes = hiddennodes
        self.outputnodes = outputnodes
        self.learningrate = learningrate
        self.active_function = active_function
        self.gradient = gradient
        self.lambda_ = lambda_

        # 权值矩阵
        self.weights_i_h = np.random.rand(
            self.hiddennodes, self.inputnodes) - 0.5
        self.weights_h_o = np.random.rand(
            self.outputnodes, self.hiddennodes) - 0.5

    def train_sgd(self, x, y):
        """梯度下降训练"""
        train_x = np.array(x).reshape(-1, 1)
        target = np.zeros((self.outputnodes, 1)) + 0.01
        target[y, 0] = 0.99

        hiddeninputs = np.dot(self.weights_i_h, train_x)
        hiddenoutputs = self.active_function(hiddeninputs)

        outputinputs = np.dot(self.weights_h_o, hiddenoutputs)
        final_outputs = self.active_function(outputinputs)

        error = target - final_outputs

        hidden_error = np.dot(self.weights_h_o.transpose(), error)

        self.weights_h_o += self.learningrate * error * \
            np.dot(self.gradient(final_outputs), hiddenoutputs.transpose())

        self.weights_i_h += self.learningrate * hidden_error * \
            np.dot(self.gradient(hiddenoutputs), train_x.transpose())

    def fit(self, train_x, targets):
        train_x = np.array(train_x)
        for i in range(train_x.shape[0]):
            self.train_sgd(train_x[i], targets[i])

    def query(self, inputs, debug=False):
        """单个值预测"""
        inputs = np.array(inputs).reshape(-1, 1)
        hidden_input = np.dot(self.weights_i_h, inputs)
        hidden_output = self.active_function(hidden_input)

        output_input = np.dot(self.weights_h_o, hidden_output)

        final_output = self.active_function(output_input)

        if debug:
            print('predict: ', final_output)

        return np.argmax(final_output)

    def predict(self, inputs):
        """批量预测"""
        res = []
        for x in inputs:
            res.append(self.query(x))
        return res

    def __str__(self):
        return "NeuralNetwork: \ninput_nodes = {0}, hidden_nodes = {1}, \noutputnodes = {2}, learningrate = {3}".format(
            self.inputnodes, self.hiddennodes, self.outputnodes, self.learningrate
        )


test_df = np.loadtxt("mnist_test_10.csv", delimiter=",", dtype=str)
test_df

train_df = np.loadtxt("mnist_train_100.csv", delimiter=",", dtype=str)
train_df

# 用测试数据测试
def accuracy(y_true, y_pred):
    """准确度"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return sum(y_true == y_pred)/y_true.shape[0]

# 用全部数据进行训练


def get_data():
    # train_df = np.loadtxt("mnist_train.csv", delimiter=",", dtype=str)
    # test_df = np.loadtxt("mnist_test.csv", delimiter=",", dtype=str)
    global train_df, test_df
    print(train_df.shape)
    print(test_df.shape)

    train_data = train_df.astype('int')
    train_x = train_data[:, 1:]
    train_y = train_data[:, 0]
    train_x = train_x / 255 * 0.99 + 0.01

    test_data = test_df.astype('int')
    test_x = test_data[:, 1:]
    test_y = test_data[:, 0]
    test_x = test_x / 255 * 0.99 + 0.01

    return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = get_data()

NN = NeuralNetwork(784, 100, 10, 0.3)
NN.fit(train_x, train_y)
y_pred = NN.predict(test_x)
print("准确度%.2f%%" % (100*accuracy(test_y, y_pred)))

hiddennodes = [512, 256, 128]
lrs = [0.1, 0.2, 0.3]
for node in hiddennodes:
    for lr in lrs:
        NN = NeuralNetwork(784, node, 10, lr)
        NN.fit(train_x, train_y)
        y_pred = NN.predict(test_x)
        print("隐藏层节点数%d,学习率%f,准确度%.2f%%" %
              (node, lr, 100*accuracy(test_y, y_pred)))

import pickle
# 最佳参数
# 隐藏层节点数128,学习率0.100000,准确度70.00%
NN = NeuralNetwork(784, 128, 10, 0.1)

# 训练10次，每3次训练下降一次学习率
for e in range(1, 11):
    if e % 3 == 0:
        NN.learningrate /= 2
    NN.fit(train_x, train_y)
    y_pred = NN.predict(test_x)
    print("第%d次训练,准确度%.2f%%" % (e, 100*accuracy(test_y, y_pred)))
    with open('NN{}.pkl'.format(e), 'wb') as f:  # 保存模型
        pickle.dump(pickle.dumps(NN), f)
