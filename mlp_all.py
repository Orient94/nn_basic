import numpy as np

class FullConnectLayer(object):
    def __init__(self, input_num, output_num):
        self.input_num = input_num
        self.output_num = output_num
    
    def init_params(self):
        self.weight = np.random.normal(
                            loc=0,scale=0.1,size=(
                            self.input_num, self.output_num))
        self.bias = np.zeros([1, self.output_num])

    def forward(self, input):
        self.input = input
        self.output = np.matmul(self.input, 
                        self.weight) + self.bias
        return self.output
    
    def backward(self, top_diff):
        import pdb;pdb.set_trace()
        self.d_weight = np.dot(self.input.T, top_diff)
        self.d_bias = np.sum(top_diff, axis=0)
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff

    def update_param(self, lr):
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr * self.d_bias

class ReLULayer(object):

    def __init__(self):
        pass
    
    def forward(self, inputs):
        self.input = inputs
        self.output = np.maximum(0, inputs)
        return self.output
    
    def backward(self, top_diff):
        bottom_diff = top_diff
        bottom_diff[self.input<=0] = 0
        return bottom_diff

class MLP:

    def __init__(self, input_dim, hid_dim_ls, output_dim):
        self.fc1 = FullConnectLayer(input_dim, hid_dim_ls[0])
        self.relu1 = ReLULayer()
        self.fc2 = FullConnectLayer(hid_dim_ls[0], hid_dim_ls[1])
        self.relu2 = ReLULayer()
        self.fc3 = FullConnectLayer(hid_dim_ls[1], output_dim)
        self.update_parems_list = [self.fc1, self.fc2, self.fc3]
        self.init_parems()

    def init_parems(self):
        for imodel in self.update_parems_list:
            imodel.init_params()
    
    def forward(self, input):
        h1 = self.fc1.forward(input)
        h1 = self.relu1.forward(h1)
        h2 = self.fc2.forward(h1)
        self.h2 = self.relu2.forward(h2)
        self.out = self.fc3.forward(h2)
        return self.out

    def loss_backward(self, label):
        bottom_diff = (label - self.out)/label.shape[0]
        return bottom_diff
    
    def backward(self, label):
        bottom_diff = self.loss_backward(label)
        bh3 = self.fc3.backward(bottom_diff)
        bh2 = self.relu2.backward(bh3)
        bh2 = self.fc2.backward(bh2)
        bh1 = self.relu1.backward(bh2)
        bh1 = self.fc1.backward(bh1)

    def update(self, lr):
        for layer in self.update_parems_list:
            layer.update_param(lr)


if __name__ == '__main__':
    
    input_data = np.random.randint(0, 100, size=(32, 20))
    output_data = np.random.randint(0,10,size=(32,1))

    hidden_size_ls = [30,30]
    mlp = MLP(input_dim=20,hid_dim_ls=hidden_size_ls,output_dim=1)

    out = mlp.forward(input_data)
    print((out-output_data).mean())

    bottom_diff = mlp.loss_backward(output_data)
    mlp.backward(output_data)
    mlp.update(lr=1)
    out = mlp.forward(input_data)
    print((out-output_data).mean())
