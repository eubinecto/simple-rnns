import torch


def main():
    X = torch.ones(size=(10, 64))
    print(X.repeat((1, 2)).shape)


if __name__ == '__main__':
    main()
