import torch
import torch.nn as nn
 
 
class NeuralNetwork(nn.Module):
 
    def __init__(self, input_size, num_classes, list_hidden,
                 activation='relu', dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.list_hidden = list_hidden
        self.activation = activation
        self.dropout = dropout
 
    def create_network(self):
        layers = []
        sizes = [self.input_size] + list(self.list_hidden)
 
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(self.get_activation(self.activation))
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
 
        layers.append(nn.Linear(sizes[-1], self.num_classes))
        layers.append(nn.Softmax(dim=1))
        self.layers = nn.Sequential(*layers)
 
    def init_weights(self, seed=2):
        torch.manual_seed(seed)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                nn.init.constant_(module.bias, 0.0)
 
    def get_activation(self, mode='sigmoid'):
        if mode == 'tanh':
            return nn.Tanh()
        if mode == 'relu':
            return nn.ReLU()
        return nn.Sigmoid()
 
    def forward(self, x, verbose=False):
        x = x.float()
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            if verbose:
                print('Output of layer ' + str(i))
                print(x, '\n')
 
        probabilities = self.layers[-1](x)
        if verbose:
            print('Output of layer ' + str(len(self.layers) - 1))
            print(probabilities, '\n')
        return x, probabilities
 
    def predict(self, probabilities):
        return torch.argmax(probabilities, dim=1)