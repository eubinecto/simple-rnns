
import torch


def main():
    Q = torch.randn(size=(10, 30, 512))  # (N, L, H)
    K = torch.randn(size=(512, 30))  # (H, L)
    print(torch.einsum("nah,hb->nab", Q, K))
    #  (H, L) -> (N, H, L) - N만큼 (H, L)행렬을 복사.
    # d (N, L, H)  * (N, H, L) -> (N, L, L)

if __name__ == '__main__':
    main()