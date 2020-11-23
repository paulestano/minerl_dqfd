import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DQN(nn.Module):
    # def __init__(self, dtype, input_shape, num_actions):
    #     super(DQN, self).__init__()
    #     self.dtype = dtype
    #
    #     self.conv1 = nn.Conv2d(input_shape[0], 32, 5, padding=1)
    #     self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
    #     self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
    #
    #     conv_out_size = self._get_conv_output(input_shape)
    #
    #     self.lin1 = nn.Linear(conv_out_size, 256)
    #     self.lin2 = nn.Linear(256, 256)
    #     self.lin3 = nn.Linear(256, num_actions)
    #
    #     self.type(dtype)
    #
    # def _get_conv_output(self, shape):
    #     input = Variable(torch.rand(1, *shape))
    #     output_feat = self._forward_conv(input)
    #     print("Conv out shape: %s" % str(output_feat.size()))
    #     n_size = output_feat.data.view(1, -1).size(1)
    #     return n_size
    #
    # def _forward_conv(self, x):
    #     x = F.relu(F.max_pool2d(self.conv1(x), 3, 2))
    #     x = F.relu(F.max_pool2d(self.conv2(x), 3, 2))
    #     x = F.relu(F.max_pool2d(self.conv3(x), 3, 2))
    #     return x
    #
    # def forward(self, states):
    #     x = self._forward_conv(states)
    #
    #     # flattening each element in the batch
    #     x = x.view(states.size(0), -1)
    #
    #     x = F.leaky_relu(self.lin1(x))
    #     x = F.leaky_relu(self.lin2(x))
    #     return self.lin3(x)
    def __init__(self, h, w, hidden_layers, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.hidden = nn.Linear(linear_input_size, hidden_layers)
        self.head = nn.Linear(hidden_layers, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))