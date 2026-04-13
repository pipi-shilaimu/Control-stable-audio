import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, name):
        super(SimpleModel, self).__init__()
        self.layer = nn.Linear(10, 10)
        self.name = name

    def predict(self, input_data):
        return f"Predicted output for {input_data} using {self.name}"
    
    def forward(self, x):
        return self.layer(x)
    