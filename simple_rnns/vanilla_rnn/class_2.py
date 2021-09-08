"""
간단한 감성분석을 할 수 있는 간단한 RNN을 pytorch로 개발하는 것이 목표.
"""

import torch
from typing import List, Tuple
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


# 신경망을 만들고 싶다 -> torch.nn.Module을 상속받기.
class SimpleRNN(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, embed_size: int):
        """
        :param vocab_size: 말뭉치 속 고유한 단어의 개수. |V|
        :param hidden_size: rnn의 hidden vector의 차원의 크기. H
        """
        # 학습하고자하는 가중치를 정의해주면 됩니다.
        super().__init__()
        # 정수인코딩된 나열을 그대로 입력으로 넣으면, 글자의 크기를 학습하게된다.
        # 그래서 원핫인코딩 or 임베딩.
        # Embedding layer = (|V|, E)
        self.hidden_size = hidden_size
        self.Embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        # h_t = f_W(h_t-1, x_t)
        # 정무님 - Wa, Wy, ba, by
        # 우리가 배웠던 기호 - W_hh, b_hh, Wxh, b_hh
        # h_t = tanh(W_hh * h_t-1 + W_xh * x_t)
        # h_t-1 (N, H) 과 h_t (N, H) 의 차원은 동일하다.
        # 즉. h_t-1 * W_hh  =  (N, H) * (H, H) = (N, H)
        # (N, L)
        # -> embeding layer -> (N, E) = x_t
        # x_t * W_xh = (N, E) * (E, H) =  (N, H)
        self.W_hh = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)  # (A, B)
        self.W_xh = torch.nn.Linear(in_features=embed_size, out_features=hidden_size)  # (C, D)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: 정수 인코딩된 나열의 배치 [[1, 2, 3, 0, 0], [5, 3, 2, 1, 0]]  (N, L)
        :return: 마지막 시간대의 hidden state를 출력 (N, H)
        """
        # forward pass를 했을때 나오는 출력값을 계산
        # h_t = tanh(W_hh * h_t-1 + W_xh * x_t)
        # 첫번째 h_t는 무의미한 값이 들어간다.
        # H_0 = torch.zeros(size=(X.shape[0], self.hidden_size))
        # # 그 다음으로는 뭘 해야돼죠?
        # # 지금 H_0을 구했다.
        # # H_1 = tanh(H_0 * W_hh + X_1 * W_xh)
        # H_1 = torch.tanh(H_0 * self.W_hh + X[:, 0] * self.W_xh)
        # H_2 = torch.tanh(H_1 * self.W_hh + X[:, 1] * self.W_xh)
        # H_3 = torch.tanh(H_2 * self.W_hh + X[:, 2] * self.W_xh)
        # 그런데 이렇게 계속 반복을 하는 것보단, 일반화를 하는 것이 낫지 않을까?
        # # for loop
        H_t = torch.zeros(size=(X.shape[0], self.hidden_size))
        for t in range(X.shape[1]):
            H_t = torch.tanh(H_t * self.W_hh + X[:, t] * self.W_xh)
        # many to one (sentiment analysis)
        return H_t

    def training_step(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param X: 정수 인코딩된 나열의 배치 [[1, 2, 3, 0, 0], [5, 3, 2, 1, 0]]  (N, L)
        :param y: 긍/부정 레이블 [0, 1, 0, 0 , 1]
        :return: loss (1)
        """
        # 배치가 들어왔을 때, 로스를 계산하면 된다.
        H_last = self.forward(X)  # (N, L) -> (N, H)
        # 이것으로부터, 긍정 / 부정일 확률을 구하고 싶다.
        # (N, H) * ??? -> (N, 1) - 어떻게 할 수 있을까?


# 어떤 데이터로 학습할까?
# 10개만 받겠습니다.
# 긍정적인 문장 5개를 적어주세요.
DATA: List[Tuple[str, int]] = [
    # 병국님
    ("오늘도 수고했어", 1),
    # 영성님
    ("너는 할 수 있어", 1),
    # 정무님
    ("오늘 내 주식이 올랐다", 1),
    # 우철님
    ("오늘 날씨가 좋다", 1),
    # 유빈님
    ("난 너를 좋아해", 1),
    # --- 부정적인 문장 - 레이블 = 0
    ("난 너를 싫어해", 0),
    # 병국님
    ("넌 잘하는게 뭐냐?", 0),
    # 선희님
    ("너 때문에 다 망쳤어", 0),
    # 정무님
    ("오늘 피곤하다", 0),
    # 유빈님
    ("난 삼성을 싫어해", 0)
]


def main():
    # 문장만,
    sents = [
        sent
        for sent, _ in DATA
    ]
    # 레이블만,
    labels = [
        label
        for _, label in DATA
    ]
    # 정무님: sents: List[str] -> List[int]
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(sents)
    seqs = tokenizer.texts_to_sequences(sents)
    for seq in seqs:
        print(seq)

    # 병국님, 영성님 - 패딩처리가 필요하다. 문장의 길이가 다 다르므로, 규격의 통일이 필요하다.
    seqs = pad_sequences(sequences=seqs, padding='post', value=0)
    print(seqs)




if __name__ == '__main__':
    main()


