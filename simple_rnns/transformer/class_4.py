
import torch
import numpy as np


class MultiHeadSelfAttentionLayer(torch.nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, heads: int):
        super().__init__()
        # 여기가 뭘하는 곳?
        # 최적화해야하는 가중치를 정의하는 곳.
        # 최적화해야하는 가중치?
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.heads = heads
        # hiden_size가 heads로 나뉘지 않으면 오류가 뜸. (AssertError)
        assert hidden_size % heads == 0
        self.W_q = torch.nn.Linear(embed_size, hidden_size)
        self.W_k = torch.nn.Linear(embed_size, hidden_size)
        self.W_v = torch.nn.Linear(embed_size, hidden_size)
        # 스킵을 하고, forward를 정의를 하면서,  shape을 알아가면 된다.
        self.W_o = torch.nn.Linear(hidden_size, hidden_size)  #  (?,?)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L, E) - 임베딩 벡터 배치.
        :return:
        """
        # 신경망의 출력을 정의하는 부분.
        # 임베딩 벡터를 입력으로 받았을 떄, 무엇을 계산하나?
        # Q, K, V는 어떻게 구하나?
        # W_q, W_k, W_v -> 목적 = 기존에는 임베딩 벡터였던 것을, 의미를 알 수없는 hidden vector.
        # E -> H
        Q = self.W_q(X)  # (..., L, E) * (E, H) -> (N, L, H )
        K = self.W_k(X)  # (..., L, E) * (E, H) -> (N, L, H )
        V = self.W_v(X)  # (..., L, E) * (E, H) -> (N, L, H )
        # 우리가 바라는 것 ->ㄷ 나중에 유사도 행렬을 구했을 떄, (N, L, L) ->  (N, heads, L, L); 모든 N개의 문장에 대하여,  heads개의 L, L 유사도 행렬이 나오길 바란다.
        N, L, H = Q.size()
        Q = Q.reshape(N, self.heads, L, H // self.heads)  # (N, L, H) -> (N, heads, L, H / heads)
        K = K.reshape(N, self.heads, L, H // self.heads)  # (N, L, H) -> (N, heads, L, H / heads)
        V = V.reshape(N, self.heads, L, H // self.heads)  # (N, L, H) -> (N, heads, L, H / heads)
        # (N, L, E) * (E, H)
        # (E, H)를 N개 복사하여, (N, E, H)를 만들고,
        # (N, L, E) * (N, E, H) -> (..., L, E) * (..., E, H)-> (N, L, H)
        # batched matrix multiplication.
        Out = self.scaled_dot_attention(Q, K, V)  # (N, heads, L, H / heads) -> (N, L, H)
        return Out

    def scaled_dot_attention(self,
                             Q: torch.Tensor,
                             K: torch.Tensor,
                             V: torch.Tensor) -> torch.Tensor:
        """
        1. Q * K^T = 모든 쿼리와 모든 키사이의 유사도를 계산.
        :param Q: (N, heads, L, H / heads)
        :param K: (N, heads, L, H / heads)
        :param V: (N, heads, L, H / heads)
        :return: (N, L, H)
        """
        # 이젠 뭘하죠?
        # Q * K^T : 모든 쿼리와 모든 키사이의 유사도를 계산.
        # self-attention - 쿼리의 개수 = 키의 개수.
        # L, L, 즉 유사도행렬은 정방행렬이 된다.
        Q = Q / np.sqrt(self.hidden_size)
        K = K / np.sqrt(self.hidden_size)
        # Out_ = torch.einsum("nah,nbh->nab", Q, K)  # (..., L, H) * (..., L, H) -> (..., L, H) * (..., H, L) = (N, L, L)
        # heads = e
        # H / heads = x
        Heads = torch.einsum("neax,nebx->neab", Q, K)  # (N, heads, L, H / heads), (N, heads, L, H / heads) -> (N, heads, L, L)
        # 다음으로는 뭘하나?
        # Out_ = Out_ / np.sqrt(self.hidden_size)  # 이렇게 스케일링을 하면, 사실 논리적인 오류가 있다. 소 잃고 외양간 고치는 격.
        # 내적을 계산하기전에, 미리 스케일링을 해야한다.
        Heads = torch.softmax(Heads, dim=2)  # (N, heads, L, L)
        # 다음은?
        # 가중평균 계산.
        # softmax(...) * V
        # selfattentionlayer의 최종 출력
        # 목적 - 입력으로 들어온 V에, 맥락 정보를 더해주는 것만이 목적이므로,
        # 최종출력은 shape이 변하지 않는다.
        # Out = torch.einsum("nll,nlh->nlh", Out_, V)  # (N, L, L) * (N, L, H) -> (N, L, H)
        Heads = torch.einsum("nell,nelx->nelx", Heads, V)  # (N, heads, L, L) * (N, heads, L, H / heads) -> (N, heads, L, H / heads)
        # 헤드를 이어 붙여주기.
        Out_ = Heads.reshape(Q.shape[0], Q.shape[2], self.hidden_size)  # (N, heads, L, H / heads) -> (N, L, H)
        Out = self.W_o(Out_)  # (N, L, H) * (H, H) -> (N, L, H)
        return Out


class EncoderLayer(torch.nn.Module):
    def __init__(self, embed_size: int, hidden_size: int):
        super().__init__()
        # 인코더 레이어에는 학습해야하는 가중치 (신경망)이
        # 무엇이 있나?
        self.sa = MultiHeadSelfAttentionLayer(embed_size, hidden_size)
        self.norm_1 = torch.nn.LayerNorm(hidden_size)
        self.ffn = torch.nn.Linear(..., ...)
        self.norm_2 = torch.nn.LayerNorm(hidden_size)

    def forward(self, X_embed: torch.Tensor) -> torch.Tensor:
        """
        :param X_embed : (N, L, E)
        :return:
        """
        # 출력  +"add" + 입력
        # 대철님 -  feedforward + add norm 부분이 residual인가요?
        # 네 맞습니다!
        Out_ = self.sa(X_embed) + X_embed  # residual connection 1
        Out_ = self.norm_1(Out_)  # layer norm 1
        Out_ = self.ffn(Out_) + Out_  # residual connection 2
        Out_ = self.norm_2(Out_)  # layer norm 2
        return Out_


class Transformer(torch.nn.Module):
    def __init__(self, vocab_size: int, embed_size: int,  hidden_size: int, depth: int):
        super().__init__()
        self.token_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        # self.encoder_layer = EncoderLayer(embed_size, hidden_size)
        # # 인코더 레이어를 어떻게 더 쌓을 수 있을까?
        # self.encoder_layer_2 = EncoderLayer(embed_size, hidden_size)
        # # torch.nn.Sequential이라는 함수를 사용.
        self.encoders = torch.nn.Sequential(
            *[EncoderLayer(embed_size, hidden_size) for _ in range(depth)]
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L)
        :return:
        """
        X_embed = self.token_embeddings(X)  # (N, L) -> (N, L, E)
        Out_ = self.encoders(X_embed)  # (N, L, E) -> (???)


def main():
    print(...)


if __name__ == '__main__':
    main()