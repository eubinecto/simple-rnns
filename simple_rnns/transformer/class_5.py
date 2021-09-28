"""
sentiment analysis with Transformer
"""
import numpy as np
import torch


class SelfAttentionLayer(torch.nn.Module):
    # 적어도 두개의 methods
    def __init__(self, embed_size: int, hidden_size: int):
        super().__init__()
        # 1 가중치를 정의하는 곳
        # 학습하는 해야하는 가중치???
        self.hidden_size = hidden_size
        self.W_q = torch.nn.Linear(embed_size, hidden_size)
        self.W_k = torch.nn.Linear(embed_size, hidden_size)
        self.W_v = torch.nn.Linear(embed_size, hidden_size)

    # 2. 신경망의 출력을 정의하 곳
    def forward(self, X_embed: torch.Tensor) -> torch.Tensor:
        """

        X  = [
          [[..난.], [.널.], [.사랑해.]],
          [[.안.], [.녕.], [.0....]],
          [[.아.], [.아.], [.0....]]
        ]
        :param X_embed: (N=배치의 크기, L=문장의 길이, E=임베딩 벡터의 크기)
        :return:
        """
        # 계산해야하는 것?
        Q = self.W_q(X_embed)  # (N, L, E) * (?,?) -> (N, L, H)
        # (N, L, E) * (?, ?)
        # (?, ?)를 N개를 복사해서 3차원 행렬을 만든다.
        # (?, ?) -> (N, ?, ?)
        # (N, L, E) * (N, ?, ?) = (..., L, E) * (..., ?=E, ?=H) -> (..., L, H)
        K = self.W_k(X_embed)  # (N, L, E) * ?=E, ?=H)) -> (N, L, H)
        V = self.W_v(X_embed)  # (N, L, E) * ?=E, ?=H)) -> (N, L, H)
        Out = self.scaled_dot_product_attention(Q, K, V)
        return Out

    def scaled_dot_product_attention(self,
                                     Q: torch.Tensor,
                                     K: torch.Tensor,
                                     V: torch.Tensor) -> torch.Tensor:
        """
        :param Q: (N, L, H)
        :param K: (N, L, H)
        :param V: (N, L, H)
        :return:
        """
        # 이제는 뭘해야하나?
        # 1. q-k 유사도 게산을 위해, Q * K^T
        # batched matrix multiplication
        Q /= np.sqrt(self.hidden_size)
        K /= np.sqrt(self.hidden_size)
        Out_ = torch.einsum("nah,nbh->nab", Q, K)  # (N, L, H), (N, L, H) ->  (N, L, H)  * (N, H, L)  -> (N, L, L)
        # 다음으로는 뭘해야하나? -> 스케일링
        # Out_ = Out_ / np.sqrt(self.hidden_size)  # 여기에는 논리적인 오류가 있다. 소 잃고 외양간 고치는 격.
        # 다음으로는 softmax로 유사도 행렬을 확룰값으로
        Out_ = torch.softmax(Out_, dim=1)  # (N, L, L)
        # 다음으로는 뭘 계산?
        #  softmax(...) * V
        # 목적? -
        Out = torch.einsum("nll,nlh->nlh", Out_, V)  # (N, L, L) * (N, L,H) -> (N, L , H)
        return Out


def main():
    pass

if __name__ == '__main__':
    main()