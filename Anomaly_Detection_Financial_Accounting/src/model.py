import torch
import torch.nn as nn



"""
Now, let's start implementing an AENN by first implementing the encoder network using PyTorch.
For the encoder, we aim to implement a network consisting of nine fully-connected layers.
Furthermore, the encoder is specified by the following number of neurons per layer: "618-256-128-64-32-16-8-4-3".
Meaning the first layer consists of 618 neurons (specified by the dimensionality of our input data),
the second layer of 256 neurons and the subsequent layers of 128, 64, 32, 16, 8, 4 and 3 neurons respectively.

Some elements of the encoder network code below should be given particular attention:

self.encoder_Lx: defines the linear transformation of the layer applied to the incoming input:  Wx+b .
nn.init.xavier_uniform: inits the layer weights using a uniform distribution according to [9].
self.encoder_Rx: defines the non-linear transformation of the layer:  σ(⋅) .
self.dropout: randomly zeros some of the elements of the input tensor with probability  p  according to [8].
We use "Leaky ReLUs" as introduced by Xu et al. in [7] to avoid "dying" non-linearities and to speed up training convergence.
Leaky ReLUs allow a small gradient even when a particular neuron is not active.
In addition, we include the "drop-out" probability, as introduced by [8],
which defines the probability rate for each neuron to be set to zero at a forward pass to prevent the network from overfitting.
However we explore its effect on the model later in the exercise section of the lab.
Initially, we set the dropout probability of each neuron to  p=0.0  (0%), meaning that none of the neuron activiations will be set to zero.
"""

class AutoEncoder(nn.Module):

    def __init__(self,in_features):
        
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
                                    nn.Linear(in_features=in_features,out_features=512,bias=True),
                                    nn.LeakyReLU(negative_slope=0.4,inplace=True),
                                    nn.Dropout(p=0.3),

                                    nn.Linear(in_features=512,out_features=256,bias=True),
                                    nn.LeakyReLU(negative_slope=0.4,inplace=True),
                                    nn.Dropout(p=0.3),

                                    nn.Linear(in_features=256,out_features=128,bias=True),
                                    nn.LeakyReLU(negative_slope=0.4,inplace=True),
                                    nn.Dropout(p=0.3),

                                    nn.Linear(in_features=128,out_features=64,bias=True),
                                    nn.LeakyReLU(negative_slope=0.4,inplace=True),
                                    nn.Dropout(p=0.3),

                                    nn.Linear(in_features=64,out_features=32,bias=True),
                                    nn.LeakyReLU(negative_slope=0.4,inplace=True),
                                    nn.Dropout(p=0.3),

                                    nn.Linear(in_features=32,out_features=16,bias=True),
                                    nn.LeakyReLU(negative_slope=0.4,inplace=True),
                                    nn.Dropout(p=0.3),

                                    nn.Linear(in_features=16,out_features=8,bias=True),
                                    nn.LeakyReLU(negative_slope=0.4,inplace=True),
                                    nn.Dropout(p=0.3),

                                    nn.Linear(in_features=8,out_features=4,bias=True),
                                    nn.LeakyReLU(negative_slope=0.4,inplace=True),
                                    nn.Dropout(p=0.3),

                                    nn.Linear(in_features=4,out_features=3,bias=True),
                                    nn.LeakyReLU(negative_slope=0.4,inplace=True),
                                    # nn.Dropout(p=0.3)
        )

        self.decoder = nn.Sequential(
                                    nn.Linear(in_features=3,out_features=4,bias=True),
                                    nn.LeakyReLU(negative_slope=0.4,inplace=True),
                                    nn.Dropout(p=0.3),

                                    nn.Linear(in_features=4,out_features=8,bias=True),
                                    nn.LeakyReLU(negative_slope=0.4,inplace=True),
                                    nn.Dropout(p=0.3),

                                    nn.Linear(in_features=8,out_features=16,bias=True),
                                    nn.LeakyReLU(negative_slope=0.4,inplace=True),
                                    nn.Dropout(p=0.3),

                                    nn.Linear(in_features=16,out_features=32,bias=True),
                                    nn.LeakyReLU(negative_slope=0.4,inplace=True),
                                    nn.Dropout(p=0.3),

                                    nn.Linear(in_features=32,out_features=64,bias=True),
                                    nn.LeakyReLU(negative_slope=0.4,inplace=True),
                                    nn.Dropout(p=0.3),

                                    nn.Linear(in_features=64,out_features=128,bias=True),
                                    nn.LeakyReLU(negative_slope=0.4,inplace=True),
                                    nn.Dropout(p=0.3),

                                    nn.Linear(in_features=128,out_features=256,bias=True),
                                    nn.LeakyReLU(negative_slope=0.4,inplace=True),
                                    nn.Dropout(p=0.3),

                                    nn.Linear(in_features=256,out_features=512,bias=True),
                                    nn.LeakyReLU(negative_slope=0.4,inplace=True),
                                    nn.Dropout(p=0.3),

                                    nn.Linear(in_features=512,out_features=in_features,bias=True),
                                    nn.LeakyReLU(negative_slope=0.4,inplace=True),
                                    # nn.Dropout(p=0.3)
                                    
                                    )

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
    


