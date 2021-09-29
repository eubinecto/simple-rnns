import torch


def main():
    max_length = 30
    indices = torch.arange(30)
    print(indices)  # (1, L)
    # (1, L) -> (N, L)
    N = 10
    print(indices.expand(10, max_length))
    # print(indices.reshape(10, max_length))  # 이건 재배치 용도, 그래서 이 코드는 오류


if __name__ == '__main__':
    main()
