import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    def __init__(self,vector_size,labels):
        super(LinearModel, self).__init__()
        self.linermodel = nn.Sequential(
							nn.Linear(vector_size, 1024),
							nn.ReLU(),
							nn.Linear(1024, 128),
							nn.ReLU(),
                            nn.Linear(128, labels),
                            # nn.ReLU(),
                            # nn.Linear(64, labels),
							)
    def forward(self, inputs):
        outputs = self.linermodel(inputs)
        return outputs