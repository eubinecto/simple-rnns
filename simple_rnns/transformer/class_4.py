
import torch
import numpy as np


class SelfAttentionLayer(torch.nn.Module):
    def __init__(self, embed_size: int, hidden_size: int):
        super().__init__()
        # 여기가 뭘하는 곳?
        # 최적화해야하는 가중치를 정의하는 곳.
        # 최적화해야하는 가중치?
        self.hidden_size = hidden_size
        self.W_q = torch.nn.Linear(embed_size, hidden_size)
        self.W_k = torch.nn.Linear(embed_size, hidden_size)
        self.W_v = torch.nn.Linear(embed_size, hidden_size)
        # 스킵을 하고, forward를 정의를 하면서,  shape을 알아가면 된다.

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
        Q = self.W_q(X)  # (..., L, E) * (?=E, ?=H) -> (..., L, H)
        K = self.W_k(X)  # (..., L, E) * (?, ?) -> (..., L, H)
        V = self.W_v(X)  # (..., L, E) * (?, ?) -> (..., L, H)
        # (N, L, E) * (E, H)
        # (E, H)를 N개 복사하여, (N, E, H)를 만들고,
        # (N, L, E) * (N, E, H) -> (..., L, E) * (..., E, H)-> (N, L, H)
        # batched matrix multiplication.
        Out = self.scaled_dot_attention(Q, K, V)
        return Out

    def scaled_dot_attention(self,
                             Q: torch.Tensor,
                             K: torch.Tensor,
                             V: torch.Tensor) -> torch.Tensor:
        """
        1. Q * K^T = 모든 쿼리와 모든 키사이의 유사도를 계산.
        :param Q: (N, L, H)
        :param K: (N, L, H)
        :param V: (N, L, H)
        :return:
        """
        # 이젠 뭘하죠?
        # Q * K^T : 모든 쿼리와 모든 키사이의 유사도를 계산.
        # self-attention - 쿼리의 개수 = 키의 개수.
        # L, L, 즉 유사도행렬은 정방행렬이 된다.
        Q = Q / np.sqrt(self.hidden_size)
        K = K / np.sqrt(self.hidden_size)
        Out_ = torch.einsum("nah,nbh->nab", Q, K)  # (..., L, H) * (..., L, H) -> (..., L, H) * (..., H, L) = (N, ?=L , ?=
        # 다음으로는 뭘하나?
        # Out_ = Out_ / np.sqrt(self.hidden_size)  # 이렇게 스케일링을 하면, 사실 논리적인 오류가 있다. 소 잃고 외양간 고치는 격.
        # 내적을 계산하기전에, 미리 스케일링을 해야한다.
        Out_ = torch.softmax(Out_, dim=1)  # (N, L, L)
        # 다음은?
        # 가중평균 계산.
        # softmax(...) * V
        # selfattentionlayer의 최종 출력
        # 목적 - 입력으로 들어온 V에, 맥락 정보를 더해주는 것만이 목적이므로,
        # 최종출력은 shape이 변하지 않는다.
        Out = torch.einsum("nll,nlh->nlh", Out_, V)  # (N, L, L) * (N, L, H) -> (N, L, H)
        return Out



class Transformer(torch.nn.Module):
    pass



def main():
    print(...)

if __name__ == '__main__':
    main()