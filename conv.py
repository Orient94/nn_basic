import numpy as np

class conv(object):
    def __init__(self, out_channel, kernel_size):
        
        self.kernel_size = kernel_size
        self.out_channel = out_channel
        self.init_params()

    def init_params(self):
        self.kernel_filters = np.random.randint( 
                                0,10, size=self.kernel_size)

    def iterate_regions(self, image):

        h,w = image.shape

        for i in range(h - self.kernel_size[0]+1):
            for j in range(w - self.kernel_size[1]+1):
                im_region = image[i:i+self.kernel_size[0], 
                                  j:j+self.kernel_size[1]]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        h, w = input.shape
        output = np.zeros((
                        h - self.kernel_size[0]+1, 
                        w - self.kernel_size[0]+1))
        for im_region, i,j in self.iterate_regions(input):
            output[i,j] = np.sum(im_region * self.kernel_filters)
        return output

    def backward(self, d_L_d_out):
        # d_L_d_out: the loss gradient for this layer's outputs
        # 初始化一组为 0 的 gradient，3x3x8

        d_L_d_filters = np.zeros(self.kernel_filters.shape)
 
        # im_region，一个个 3x3 小矩阵
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                # 按 f 分层计算，一次算一层，然后累加起来
                # d_L_d_filters[f]: 3x3 matrix
                # d_L_d_out[i, j, f]: num
                # im_region: 3x3 matrix in image
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
        self.d_L_d_filters = d_L_d_filters
    
    def update(self, lr):
        self.kernel_filters -= lr * self.d_L_d_filters


class convMulOut:
    def __init__(self, out_channels, kernal_size):
        self.kernal_size = kernal_size
        self.out_channels = out_channels
        self.init_param()

    def init_param(self):
        self.kernal = [np.random.normal(
                            loc=0,scale=0.1,
                            size=self.kernal_size) 
                            for _ in range(self.out_channels)]

    def iters_regions(self, inputs):
        h, w = inputs.shape 

        for i in range(h - self.kernal_size[0]+1):
            for j in range(w - self.kernal_size[1]+1):
                iregion = inputs[i : i+self.kernal_size[0],
                                 j : j+self.kernal_size[1]]
                yield i, j, iregion
    
    def forward(self, inputs):
        h, w = inputs.shape
        outputs = []
        for i in range(self.out_channels):
            ioutputs = np.zeros((h - self.kernal_size[0]%2-1,
                                w - self.kernal_size[1]%2-1))
            for m, n, iregion in self.iters_regions(inputs):
                ioutputs[m,n] = np.sum(iregion * self.kernal[i])
            outputs.append(ioutputs)
        return np.array(outputs)


class convMdl:
    def __init__(self):
        
        self.conv1 = conv(out_channel=1, kernel_size=(3,3))
        self.update_layer_list = [self.conv1]

    def init_para(self):
        for imodel in self.update_layer_list:
            imodel.init_params()
    
    def forward(self, inputs):
        out = self.conv1(inputs)
        return out

    def backward(self, last_loss):
        self.conv1.backward(last_loss)
    
    def update(self, lr):
        for imodel in self.update_layer_list:
            imodel.update(lr)


if __name__ == '__main__':

    inputs = np.random.randn(28,28)
    model = conv(2, (3,3))
    out = model.forward(inputs)
    print(out.shape)

    model = convMulOut(2, (3,3))
    outs = model.forward(inputs)
    print(outs.shape)