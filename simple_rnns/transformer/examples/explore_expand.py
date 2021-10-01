import torch


def main():
    max_length = 30
    indices = torch.arange(max_length)
    print(indices)  # (1, L)
    N = 50
    print(indices.expand(50, max_length))


if __name__ == '__main__':
    main()
