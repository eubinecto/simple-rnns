
import torch


def main():
    Q = torch.randn(size=(30, 512))
    K = torch.randn(size=(30, 512))
    print(torch.einsum("ah,bh->ab", Q, K))


if __name__ == '__main__':
    main()