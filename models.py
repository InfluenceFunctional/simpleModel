from utils import *

class param_activation(nn.Module): # a better (pytorch-friendly) implementation of activation as a linear combination of basis functions
    def __init__(self, n_basis, span, channels, freezeTag, *args, **kwargs):
        super(param_activation, self).__init__(*args, **kwargs)

        self.channels, self.n_basis = channels, n_basis
        # define the space of basis functions
        self.register_buffer('dict', torch.linspace(-span, span, n_basis)) # positive and negative values for Dirichlet Kernel
        #self.register_buffer('dict', torch.linspace(1, span, n_basis + 1)) # positive values for ReLU kernel

        # define module to learn parameters
        # 1d convolutions allow for grouping of terms, unlike nn.linear which is always fully-connected.
        # #This way should be fast and efficient, and play nice with pytorch optim
        self.linear = nn.Conv1d(channels * n_basis, channels, kernel_size=(1,1), groups=int(channels), bias=False)

        if freezeTag == True:
            self.linear.weight.requires_grad = False # turn off learning for activation layer

        #nn.init.normal(self.linear.weight.data, std=0.1)

        self.alpha = nn.Parameter(torch.randn(10),requires_grad=True)

        self.eps = 1e-16

    def kernel(self, x):
        # x has dimention batch, features, y, x
        # must return object of dimension batch, features, y, x, basis
        x = x.unsqueeze(2)
        if len(x)==2:
            x = x.reshape(2,self.channels,1)

        #return (torch.sin(x * (self.dict+0.5))+self.eps)/(2 * np.pi * torch.sin(x/2) + self.eps) # dirichlet kernel
        return F.relu(x) / (self.dict)
        #return torch.exp(-(x - self.dict) ** 2)
        #return torch.sin(x * self.dict)/x

    def forward(self, x):
        x = self.kernel(x).unsqueeze(-1).unsqueeze(-1) # run activation, output shape batch, features, y, x, basis
        x = x.reshape(x.shape[0],x.shape[1]*x.shape[2],x.shape[3],x.shape[4]) # concatenate basis functions with filters
        x = self.linear(x).squeeze(-1).squeeze(-1) # apply linear coefficients and sum

        #y = torch.zeros((x.shape[0], self.channels, x.shape[-2], x.shape[-1])).cuda() #initialize output
        #for i in range(self.channels):
        #    y[:,i,:,:] = self.linear[i](x[:,i,:,:,:]).squeeze(-1) # multiply coefficients channel-wise (probably slow)

        return x


class Activation(nn.Module):
    def __init__(self, activation_func, filters, *args, **kwargs):
        super().__init__()
        if activation_func == 'relu':
            self.activation = F.relu
        elif activation_func == 'parametric':
            self.activation = param_activation(n_basis=20, span=4, channels=filters, freezeTag=False)

    def forward(self, input):
        return self.activation(input)


class MLP(nn.Module):
    def __init__(self,params):
        super(MLP,self).__init__()
        # initialize constants and layers

        if True:
            act_func = 'relu'
        #elif params['activation']==2:
        #    act_func = 'kernel'

        self.inputLength = params['input length']
        self.layers = params['model layers']
        self.filters = params['model filters']

        # build input and output layers
        self.initial_layer = nn.Linear(self.inputLength, self.filters) # layer which takes in our sequence
        self.activation1 = Activation(act_func,self.filters)
        self.output_layer = nn.Linear(self.filters, 1)

        # build hidden layers
        self.lin_layers = []
        self.activations = []
        #self.norms = []

        for i in range(self.layers):
            self.lin_layers.append(nn.Linear(self.filters,self.filters))
            self.activations.append(Activation(act_func, self.filters))
            #self.norms.append(nn.BatchNorm1d(self.filters))

        # initialize module lists
        self.lin_layers = nn.ModuleList(self.lin_layers)
        self.activations = nn.ModuleList(self.activations)
        #self.norms = nn.ModuleList(self.norms) # optional normalization layer


    def forward(self, x):
        x = self.activation1(self.initial_layer(x)) # apply linear transformation and nonlinear activation
        for i in range(self.layers):
            x = x + (self.activations[i](self.lin_layers[i](x)))
            #x = self.norms[i](x)

        x = self.output_layer(x) # linear transformation to output
        return x
