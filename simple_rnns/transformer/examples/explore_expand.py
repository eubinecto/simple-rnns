import torch


def main():
    max_length = 30
    indices = torch.arange(max_length)
    print(indices)  # (1, L)
    N = 30
    print(indices.expand(N, max_length))


if __name__ == '__main__':
    main()
