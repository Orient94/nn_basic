import numpy as np
from params import param

class FullConnectedLayer(object):

    def __init__(self, num_input, num_output):
        """输入输出的维度"""
        self.num_input = num_input
        self.num_output = num_output

    def init_param(self):
        self.weight = np.random.normal(
                        loc=0.0, scale=0.1, 
                        size=(
                        self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])

    def forward(self, input):
        self.input = input
        self.output = np.matmul(input, self.weight) + self.bias
        return self.output

    def backward(self, top_diff):
        self.d_weight = np.dot(self.input.T, top_diff)
        self.d_bias = np.sum(top_diff, axis=0)
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff

    def update_param(self, lr):
        self.weight = self.weight - lr*self.d_weight
        self.bias = self.bias - lr*self.d_bias
    
    def load_param(self, weight, bias):
        self.weight = weight
        self.bias = bias 
    
    def save_param(self):
        return self.weight, self.bias

class ReLULayer(object):
    def __init__(self):
        print('\tReLU layer.')
    def forward(self, input):  # 前向传播的计算
        self.input = input
        # TODO：ReLU层的前向传播，计算输出结果
        output = np.maximum(0, input)
        return output
    def backward(self, top_diff):  # 反向传播的计算
        # TODO：ReLU层的反向传播，计算本层损失
        bottom_diff = top_diff
        bottom_diff[self.input<0] = 0
        return bottom_diff


class SoftmaxLossLayer(object):
    def __init__(self):
        pass
    
    def forward(self, input):
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1)[:,np.newaxis]
        return self.prob
    
    def get_loss(self, label):
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss
    
    def backward(self):
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff

class RMSELayer(object):
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output
    
    def init_param(self):
        self.weight = np.random.normal(
                        loc=0.0, scale=0.1, 
                        size=(
                        self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])

    def forward(self, input):
        self.input = input
        self.output = np.matmul(self.input, self.weight) + self.bias
        return self.output

    def get_loss(self, label):
        loss = ((self.output-label) ** 2).mean()
        return loss
    
    def backward(self, loss):
        self.d_weight = np.dot(self.input.T, loss)
        self.d_bias = np.sum(loss, axis=0)
        bottom_diff = np.dot(loss, self.weight.T)
        return bottom_diff

    def update_param(self, lr):
        self.weight = self.weight - lr*self.d_weight
        self.bias = self.bias - lr*self.d_bias

class MLP(object): 
    def __init__(self, input_size, hidden_size_ls, output_size):
        self.input_size = input_size
        self.hidden_size_ls = hidden_size_ls
        self.output_size = output_size
        self.fc1 = FullConnectedLayer(
                    self.input_size, self.hidden_size_ls[0])
        self.relu1 = ReLULayer()
        self.fc2 = FullConnectedLayer(
                    self.hidden_size_ls[0], self.hidden_size_ls[1])
        self.relu2 = ReLULayer()
        self.fc3 = FullConnectedLayer(
                    self.hidden_size_ls[1], self.hidden_size_ls[2])
        self.relu3 = ReLULayer()
        self.fc4 = FullConnectedLayer(
                    self.hidden_size_ls[2], self.output_size)
        self.relu4 = ReLULayer()
        self.softmax = SoftmaxLossLayer()
        self.update_layer_list = [self.fc1, self.fc2, self.fc3, self.fc4]
        '''
        self.rmseLayer = RMSELayer(
                            self.hidden_size_ls[2], self.output_size)
        self.update_layer_list = [
            self.fc1, self.fc2, self.fc3, self.fc4, self.rmseLayer]
        '''
        self.init_model()

    def init_model(self):
        for layer in self.update_layer_list:
            layer.init_param()

    def forward(self, input):  # 神经网络的前向传播
        # TODO：神经网络的前向传播
        
        h1 = self.fc1.forward(input)
        h1 = self.relu1.forward(h1)
        h2 = self.fc2.forward(h1)
        h2 = self.relu2.forward(h2)
        h3 = self.fc3.forward(h2)
        h3 = self.relu3.forward(h3)
        h4 = self.fc4.forward(h3)
        #prob = self.rmseLayer.forward(h4)
        prob = self.softmax.forward(h4)
        return prob
    
    def backward(self, dloss):  # 神经网络的反向传播
        # TODO：神经网络的反向传播
        #dloss = self.softmax.backward()
        dh4 = self.fc4.backward(dloss)
        dh3 = self.relu3.backward(dh4)
        dh3 = self.fc3.backward(dh3)
        dh2 = self.relu2.backward(dh3)
        dh2 = self.fc2.backward(dh2)
        dh1 = self.relu1.backward(dh2)
        dh1 = self.fc1.backward(dh1)

    def update(self, lr):

        for layer in self.update_layer_list:
            layer.update_param(lr)


if __name__ == '__main__':
    

    import pdb;pdb.set_trace()
    input_data = np.random.randint(0, 100, size=(32, 20))
    output_data = np.random.randint(0,10,size=(32,))

    hidden_size_ls = [30,30,30,30]
    mlp = MLP(input_size=20,hidden_size_ls=hidden_size_ls,output_size=10)

    out = mlp.forward(input_data)
    loss = mlp.softmax.get_loss(output_data)
    dloss = mlp.softmax.backward()
    print(dloss)
    mlp.backward(dloss)

    
    mlp.update(lr=1)
    out = mlp.forward(input_data)
    loss = mlp.rmseLayer.get_loss(output_data)
    print(loss)