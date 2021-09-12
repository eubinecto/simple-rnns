import torch


def main():
    A = torch.ones(size=(10, 3, 4))
    B = torch.ones(size=(10, 4))
    # batched matrix-vector multiplication.
    R = torch.einsum("nlh,nh->nl", A, B)
    print(R)
    print(type(float("-inf")))

if __name__ == '__main__':
    main()