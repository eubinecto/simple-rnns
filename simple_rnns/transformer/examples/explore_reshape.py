
import torch

def main():
    N = 10
    L = 30
    H = 512
    num_heads = 8
    Q = torch.ones(size=(N, L, H)) # N * L * H
    print(Q.shape)
    assert H % num_heads == 0
    Q = Q.reshape(N, num_heads, L, H // num_heads)  # N * H / num_heads * num_heads * H
    print(Q.shape)
    # Q = Q.reshape()

if __name__ == '__main__':
    main()
