
import torch
from torch.nn import functional as F


def main():
    y_hat = torch.Tensor([0, 0, 1])
    y = torch.FloatTensor([0, 0, 1])
    loss = F.binary_cross_entropy(y_hat, y)
    print(loss)

if __name__ == '__main__':
    main()
