

import torch


def main():
    H = torch.zeros(size=(10, 12))  # (N, H)
    X = torch.ones(size=(10, 8))  # (N, E)
    print(torch.cat([H, X], dim=1))  # (N, H + E)


if __name__ == '__main__':
    main()
