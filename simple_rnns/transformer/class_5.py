"""
sentiment analysis with Transformer
"""
import numpy as np
import torch


class MultiHeadSelfAttentionLayer(torch.nn.Module):
    # 적어도 두개의 methods
    def __init__(self, embed_size: int, hidden_size: int, heads: int):
        super().__init__()
        # 1 가중치를 정의하는 곳
        # 학습하는 해야하는 가중치???
        self.heads = heads
        self.hidden_size = hidden_size
        assert hidden_size % self.heads == 0
        self.W_q = torch.nn.Linear(embed_size, hidden_size)
        self.W_k = torch.nn.Linear(embed_size, hidden_size)
        self.W_v = torch.nn.Linear(embed_size, hidden_size)
        self.W_o = torch.nn.Linear(hidden_size, hidden_size)

    # 2. 신경망의 출력을 정의하 곳
    def forward(self, X_embed: torch.Tensor) -> torch.Tensor:
        """

        X  = [
          [[..난.], [.널.], [.사랑해.]],
          [[.안.], [.녕.], [.0....]],
          [[.아.], [.아.], [.0....]]
        ]
        :param X_embed: (N=배치의 크기, L=문장의 길이, E=임베딩 벡터의 크기)
        :return:  (N, L, H)
        """
        # 계산해야하는 것?
        Q = self.W_q(X_embed)  # (N, L, E) * (?,?) -> (N, L, H)
        # (N, L, E) * (?, ?)
        # (?, ?)를 N개를 복사해서 3차원 행렬을 만든다.
        # (?, ?) -> (N, ?, ?)
        # (N, L, E) * (N, ?, ?) = (..., L, E) * (..., ?=E, ?=H) -> (..., L, H)
        K = self.W_k(X_embed)  # (N, L, E) * ?=E, ?=H)) -> (N, L, H)
        V = self.W_v(X_embed)  # (N, L, E) * ?=E, ?=H)) -> (N, L, H)
        # single head - (N, L, H) -> (N, Heads, L, H / Heads)
        N, L, H = Q.size()
        # 성분을 재배치, 단 heads차원을 추가.
        Q = Q.reshape(N, self.heads, L, self.hidden_size // self.heads)
        K = K.reshape(N, self.heads, L, self.hidden_size // self.heads)
        V = V.reshape(N, self.heads, L, self.hidden_size // self.heads)
        Out_ = self.scaled_dot_product_attention(Q, K, V)  # (N, Heads, L, H / heads) -> (N, Heads, L, H / heads)
        Out_ = Out_.reshape(N, L, H)  # (N, Heads, L, H / heads) -> (N, L, H)
        Out_ = self.W_o(Out_)  # (N, L, H) * (?=H, ?=H) -> (N, L, H)
        return Out_

    def scaled_dot_product_attention(self,
                                     Q: torch.Tensor,
                                     K: torch.Tensor,
                                     V: torch.Tensor) -> torch.Tensor:
        """
        :param Q: (N, Heads, L, H / heads)
        :param K: (N, Heads, L, H / heads)
        :param V: (N, Heads, L, H / heads)
        :return: (N, Heads, L, H / heads)
        """
        # 이제는 뭘해야하나?
        # 1. q-k 유사도 게산을 위해, Q * K^T
        # batched matrix multiplication
        Q /= np.sqrt(self.hidden_size)
        K /= np.sqrt(self.hidden_size)
        # Out_ = Q @ K.reshape(K.shape[0], K.shape[2], K.shape[1])
        # Out_ = torch.einsum("nah,nbh->nab", Q, K)  # (N, L, H), (N, L, H) ->  (N, L, H)  * (N, H, L)  -> (N, L, L)
        # heads = e
        # H / heads = x
        Out_ = torch.einsum("neax,nebx->neab", Q, K)  # (N, Heads, L, H / heads), (N, Heads, L, H / heads) ->  (N, heads, L, L)
        # 다음으로는 뭘해야하나? -> 스케일링
        # Out_ = Out_ / np.sqrt(self.hidden_size)  # 여기에는 논리적인 오류가 있다. 소 잃고 외양간 고치는 격.
        # 다음으로는 softmax로 유사도 행렬을 확룰값으로
        # Out_ = torch.softmax(Out_, dim=1)  # (N, L, L)
        Out_ = torch.softmax(Out_, dim=2)  # (N, Heads, L, L)
        # 다음으로는 뭘 계산?
        #  softmax(...) * V
        # 목적? -
        # Out = torch.einsum("nll,nlh->nlh", Out_, V)  # (N, L, L) * (N, L,H) -> (N, L , H)
        Out = torch.einsum("nell,nelx->nelx", Out_, V)  # (N, Heads, L, L) * (N, Heads, L, H / heads) -> (N, Heads, L, H / heads)
        return Out


class EncoderLayer(torch.nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, heads: int):
        super().__init__()
        # 필요한 가중치 정의
        # 어떤 신경망을 학습해야하나?
        assert embed_size == hidden_size
        self.mhsa = MultiHeadSelfAttentionLayer(embed_size, hidden_size, heads)
        self.norm_1 = torch.nn.LayerNorm(hidden_size)
        self.ffn = torch.nn.Linear(hidden_size, hidden_size)
        self.norm_2 = torch.nn.LayerNorm(hidden_size)
        # 또 학습해야하는 레이어?

    def forward(self, X_embed: torch.Tensor) -> torch.Tensor:
        """
        :param X_embed: (N, L, E)
        :return: (N, L, H)
        """
        Out_ = self.mhsa(X_embed) + X_embed  # (N, L, E) -> (N, L, H). (N, L, H) + (N, L, E)
        Out_ = self.norm_1(Out_)  #  (N, L, H)
        Out_ = self.ffn(Out_) + Out_  #  (N, L, H) * (?=H, ?=H) -> (N, L, H)
        Out = self.norm_2(Out_)
        return Out


# --- hyper parameters --- #
EMBED_SIZE = 512
HIDDEN_SIZE = 512



def main():
    pass

if __name__ == '__main__':
    main()