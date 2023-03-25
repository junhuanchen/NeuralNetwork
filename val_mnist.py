
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
        self.weights_i_h = np.random.rand(self.hiddennodes, self.inputnodes) - 0.5 
        self.weights_h_o = np.random.rand(self.outputnodes, self.hiddennodes) - 0.5
        
    def train_sgd(self, x, y):
        """梯度下降训练"""
        train_x = np.array(x).reshape(-1,1)
        target = np.zeros((self.outputnodes,1)) + 0.01
        target[y,0] = 0.99
        
        hiddeninputs = np.dot(self.weights_i_h, train_x)
        hiddenoutputs = self.active_function(hiddeninputs)
        
        outputinputs = np.dot(self.weights_h_o, hiddenoutputs)
        final_outputs = self.active_function(outputinputs)
        
        error = target - final_outputs
        
        hidden_error = np.dot(self.weights_h_o.transpose(), error)
        
        self.weights_h_o += self.learningrate * error * np.dot(self.gradient(final_outputs), hiddenoutputs.transpose())
        
        self.weights_i_h += self.learningrate * hidden_error * np.dot(self.gradient(hiddenoutputs), train_x.transpose()) 
    
    def fit(self, train_x, targets):
        train_x = np.array(train_x)
        for i in range(train_x.shape[0]):
            self.train_sgd(train_x[i], targets[i])
    
    def query(self, inputs, debug=False):
        """单个值预测"""
        inputs = np.array(inputs).reshape(-1,1)
        hidden_input = np.dot(self.weights_i_h, inputs)
        hidden_output = self.active_function(hidden_input)
        
        output_input = np.dot(self.weights_h_o, hidden_output)
        
        final_output = self.active_function(output_input)

        if debug:
            print('predict: ', final_output)
            
        return np.argmax(final_output)
    
    def predict(self,inputs):
        """批量预测"""
        res = []
        for x in inputs:
            res.append(self.query(x))
        return res
    
    def __str__(self):
        return "NeuralNetwork: \ninput_nodes = {0}, hidden_nodes = {1}, \noutputnodes = {2}, learningrate = {3}".format(
            self.inputnodes, self.hiddennodes, self.outputnodes, self.learningrate
        )

import pickle

# 最佳模型，载入最佳模型
with open('NN10.pkl','rb') as f:
    b_data = pickle.load(f)
    net_model = pickle.loads(b_data)
print(net_model)

# 使用 PIL 生成一张 28*28 的图，并绘制数字，转成 784*1 的向量，用最佳模型预测
from PIL import Image, ImageDraw, ImageFont

# 如果 imgs 目录存在则删除重新创建
import os
if not os.path.exists('./imgs'):
    os.mkdir('./imgs')
else:
    for f in os.listdir('./imgs'):
        os.remove('./imgs/{}'.format(f))

for f, s in [('sxsz.ttf', 22), ('sans.ttf', 22)]:
    for i in range(10):
        img = Image.new('L', (28,28), 0)
        font = ImageFont.truetype('./resc/%s' % f, s)
        img_draw = ImageDraw.Draw(img)
        img_draw.text((7,0), str(i), fill=255, font=font)
        
        tmp = np.array(img).reshape(1,784)
        tmp = tmp / 255 * 0.99 + 0.01
        print(tmp.shape)

        id = net_model.query(tmp)
        print(id)

        img.save('./imgs/{}_{}_{}.png'.format(f, i, id))
