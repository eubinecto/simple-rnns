import torch


def main():
    A = torch.randn(size=(10, 3, 4))
    B = torch.randn(size=(10, 4))
    # batched matrix-matrix multiplication.
    R = torch.einsum("nlh,nh->nl", A, B)
    print(R)
    R = torch.einsum("nlh,hn->nl", A, B.T)
    # 알아서 차원을 정렬해준다. (reduces over the overlapping dimension).
    print(R)
    print(type(float("-inf")))


if __name__ == '__main__':
    main()