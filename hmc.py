#!/usr/bin/env python3
"""Binary for Hierarchical multi-label classification"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_integer('mlp_hidden_size', 1024, 
                     'Number of output dimensions for MLP')
flags.DEFINE_float('Lambda', 0.5, 
                     'loss parameter for L2')
flags.DEFINE_float('Beta', 0.5, 
                     'loss parameter for L2-L1')
flags.DEFINE_float('mask_value', -100, 
                     'mask_value for L2 classification')

class HMC(nn.Module):
    """A classifier for Hierarchical multi-label classification"""

    def __init__(self, 
                 feature_size,
                 L1_labels_num,
                 L2_labels_num,
                 L12_table,
                 mlp_hidden_size=None,
                 mask_value=None):
        """Construct a classifier for HMC

        Args:
            feature_size: feature size in input_tensor
            mlp_hidden_size: Number of output dimensions for MLP
            L1_labels_num: Number of labels on the first level
            L2_labels_num: Number of labels on the second level
            L12_table: A list of some lists.
                       For example, L12_table[i][j] is a number, which means 
                       that L12_table[i][j](L2 label number) belongs to i(L1 label)
            mask_value: a float number which make exp(mask_value) close to zero
        """

        super(HMC, self).__init__()

        self.mlp_hidden_size = mlp_hidden_size or FLAGS.mlp_hidden_size
        self.mask_value = mask_value or FLAGS.mask_value
        self.feature_size = feature_size
        self.L1_labels_num = L1_labels_num
        self.L2_labels_num = L2_labels_num
        self.L12_table = L12_table

        assert len(L12_table) == L1_labels_num
        assert self.check_L12_table(L12_table)

        self.fc_L1_1 = nn.Linear(self.feature_size, self.mlp_hidden_size)
        self.fc_L1_2 = nn.Linear(self.mlp_hidden_size, self.L1_labels_num)
        self.fc_L2_1 = nn.Linear(self.feature_size, self.mlp_hidden_size)
        self.fc_L2_2 = nn.Linear(2*self.mlp_hidden_size, self.L2_labels_num)
        

    def check_L12_table(self, L12_table):
        """check the Legality of L12_table"""

        L2_labels = [num for lst in L12_table for num in lst]
        assert len(L2_labels) == self.L2_labels_num
        for i in range(self.L2_labels_num):
            if i not in L2_labels:
                return False
        return True

    def forward(self,x):
        """forward computation

        Args:
            x: input_tensor, must only have 2 dims
        """

        assert len(x.shape) == 2

        L1 = F.relu(self.fc_L1_1(x))
        L2 = F.relu(self.fc_L2_1(x))
        L2 = torch.cat((L1,L2), dim=1)
        L1 = F.relu(self.fc_L1_2(L1))
        L1 = F.softmax(L1, dim=1)
        L2 = F.relu(self.fc_L2_2(L2))

        L1_label = L1.argmax(dim=1)
        mask = torch.ones_like(L2) * self.mask_value

        """for-loop may be not a good choice.
        But I don't have any other methods
        """
        for i,element in enumerate(L1_label):
            idx = element.item()
            mask[i,self.L12_table[idx]]=0

        L2 += mask
        L2 = F.softmax(L2,dim=1)

        return L1,L2

def hmc_loss(L1,L2,L1_gt,L2_gt,Lambda=None,Beta=None):
    """calculate hmc loss

    Args:
        L1: Tensor(batch*L1_labels_num) -> network output of level 1
        L2: Tensor(batch*L2_labels_num) -> network output of level 2
        L1_gt: Tensor(batch,) dtype=int -> ground truth of level 1
        L2_gt: Tensor(batch,) dtype=int -> ground truth of level 2
        Lambda: a float coefficient for L2 loss pernelty
        Beta: a float coefficient for L2-L1 loss pernelty
    """

    Lambda = Lambda or FLAGS.Lambda
    Beta = Beta or FLAGS.Beta

    batch_num = L1.shape[0]
    Y1 = L1[torch.arange(batch_num),L1_gt]
    Y2 = L2[torch.arange(batch_num),L2_gt]

    L1_loss = - Y1.log().mean()
    L2_loss = - Y2.log().mean()
    LH_loss = torch.max(Y2-Y1,torch.zeros_like(Y1)).mean()
    
    return L1_loss + Lambda * L2_loss + Beta * LH_loss 

def main(argv):
    # del argv  # Unused.
   
    # print('Running under Python {0[0]}.{0[1]}.{0[2]}'.format(sys.version_info),
    #       file=sys.stderr)
    # logging.info('echo is %s.', FLAGS.echo)
    model = HMC(1024,5,7,[[0],[2,3,4],[5],[6],[1]])
    L1,L2 = model(torch.rand(3,1024))
    L1_gt=torch.Tensor(
        [0,1,1]
    ).long()
    L2_gt=torch.Tensor(
        [0,2,4]
    ).long()
    loss = hmc_loss(L1,L2,L1_gt,L2_gt)
    print(loss)


if __name__ == '__main__':
    app.run(main)

