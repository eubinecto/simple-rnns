import torch


def main():
    A = torch.Tensor([
        [1, 2, 3],
        [4, 5, 6]
    ])
    print(torch.softmax(A, dim=1))  # 두번째 차원에 걸쳐서 정규화.
    print(torch.softmax(A, dim=0))  # 첫번째 차원에 걸쳐서 정규화



if __name__ == '__main__':
    main()