"""
감성분석을 할 수 있는 charRNN을 만드는 것.
analyser("나는 자연어처리가 좋아") -> 확률
"""
from typing import List, Tuple

import torch.nn
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from torch.nn import functional as F

DATA: List[Tuple[str, int]] = [
    # 긍정적인 문장 - 1
    ("나는 자연어처리", 1),
    ("도움이 되었으면", 1),
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
    # 다운님
    ("지금 정말 잘하고 있어", 1),
    # 민종님
    ("지금처럼만 하면 잘될거야", 1),
    ("사랑해", 1),
    ("저희 허락없이 아프지 마세요", 1),
    ("오늘 점심 맛있다", 1),
    ("오늘 너무 예쁘다", 1),
    # 다운님
    ("곧 주말이야", 1),
    # 재용님
    ("오늘 주식이 올랐어", 1),
    # 병운님
    ("우리에게 빛나는 미래가 있어", 1),
    # 재용님
    ("너는 참 잘생겼어", 1),
    # 윤서님
    ("콩나물 무침은 맛있어", 1),
    # 정원님
    ("강사님 보고 싶어요", 1),
    # 정원님
    ("오늘 참 멋있었어", 1),
    # 예은님
    ("맛있는게 먹고싶다", 1),
    # 민성님
    ("로또 당첨됐어", 1),
    # 민성님
    ("이 음식은 맛이 없을수가 없어", 1),
    # 경서님
    ("오늘도 좋은 하루보내요", 1),
    # 성민님
    ("내일 시험 안 본대", 1),
    # --- 부정적인 문장 - 레이블 = 0
    ("난 너를 싫어해", 0),
    # 병국님
    ("넌 잘하는게 뭐냐?", 0),
    # 선희님
    ("너 때문에 다 망쳤어", 0),
    # 정무님
    ("오늘 피곤하다", 0),
    # 유빈님
    ("난 삼성을 싫어해", 0),
    ("진짜 가지가지 한다", 0),
    ("꺼져", 0),
    ("그렇게 살아서 뭘하겠니", 0),
    # 재용님 - 주식이 파란불이다?
    ("오늘 주식이 파란불이야", 0),
    # 지현님
    ("나 오늘 예민해", 0),
    ("주식이 떨어졌다", 0),
    ("콩나물 다시는 안먹어", 0),
    ("코인 시즌 끝났다", 0),
    ("배고파 죽을 것 같아", 0),
    ("한강 몇도냐", 0),
    ("집가고 싶다", 0),
    ("나 보기가 역겨워", 0),  # 긍정적인 확률이 0
    # 진환님
    ("잘도 그러겠다", 0),
]



# 파이토치에서 신경망은 어떻게 구현?
# Module 을 상속하고,


# 1. __init__ 2. forward 3. training_step
class SimpleRNN(torch.nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # 학습해야하는 가중치를 정의하는 곳.
        self.E = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        # 임베딩 레이어 외에, 또 학습해야 하는 가중치?
        # 하나의 perceptron layer.
        # y = x * W^T + b
        # 가중치 행렬의 shape를 어떻게 알수 있을까?
        # 1. 일단 스킵하자.
        # 2. 행렬의 차원을 트래킹하면서 결정하면 된다.
        # 3. (A, B) * (B, C) -> (A, C).
        self.W_hh = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)  # (?=H, ?=H)
        self.W_xh = torch.nn.Linear(in_features=embed_size, out_features=hidden_size)  # (?=E, ?=H)
        self.W_hy = torch.nn.Linear(in_features=hidden_size, out_features=1)  # (?=H, ?=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L=16)
        :return:
        """
        # 어떻게 할수 있을까?
        # 모델의 최종 출력값을 정의하고 반환하는 함수.
        # H_t = f_W(H_t-1, X_t)
        # H_t = tanh(W_hh * H_t-1 + W_xh * X_t)
        # 시간대의 길이는?
        H_t = torch.zeros(size=(X.shape[0], self.hidden_size))  # (N, H)
        for time in range(X.shape[1]):  # 시간 = 0 -> L
            X_t = X[:, time]  # (N, L) -> (N, 1)
            # 정수 인코딩에 대응하는 임베딩 벡터를 가져온다
            X_t = self.E(X_t)  # (N, 1) -> (N, E)
            # self.W_hh(H_t) (N, H) * (?=H, ?=H) -> (N, H)
            # self.W_xh(X_t) (N, E) * (?=E, ?=H) -> (N, H)
            H_t = torch.tanh(self.W_hh(H_t) + self.W_xh(X_t))  # -> (N, H).
        H_last: torch.Tensor = H_t
        return H_last  # (N, H) - 사이즈가 H인 N개의 hidden vector를 출력.

    def training_step(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L)  = N개의 문장의 정수인코딩의 나열.
        :param y: (N, 1)  = N개의 문장에 대응하는 10101010101
        :return: loss (1). 하나의 실수 값. 배치 = 로스를 계산하는 단위.
        """
        # 배치에 대한 로스를 계산하는 함수.
        H_last = self.forward(X)  # (N, L) -> (N, H)
        y_pred = self.W_hy(H_last)  # (N, H) * (?=H,?=1) -> (N, 1)
        # y_pred = 범위? =[-inf, +inf]
        # 그런데 우리가 원하는 건 확률.
        y_pred = torch.sigmoid(y_pred)  # (N, 1) = N개의 문장에 대응하는 하나의 확률. 범위 = [0,1]
        # y_pred = (N, 1) or (N,)
        y_pred = torch.reshape(y_pred, y.shape)  # y_pred의 차원을 y의 차원과 동기화
        loss: torch.Tensor = F.binary_cross_entropy(input=y_pred, target=y)  # (N, 1), (N, 1) -> (N, 1)
        loss = loss.sum()  # (N, 1) -> (1)
        return loss


# hyper paramters #
EMBED_SIZE = 32
HIDDEN_SIZE = 64  # RNN에 있는 뉴런의 개수.
LR = 0.001
EPOCHS = 300

# builders #
def build_X(sents: List[str], tokenizer: Tokenizer) -> torch.Tensor:
    seqs = tokenizer.texts_to_sequences(texts=sents)
    seqs = pad_sequences(sequences=seqs, padding="post", value=0)
    return torch.LongTensor(seqs)


def build_y(labels: List[int]) -> torch.Tensor:
    return torch.FloatTensor(labels)


class Anaylser:
    def __init__(self, rnn: SimpleRNN, tokenizer: Tokenizer):
        self.rnn = rnn
        self.tokenizer = tokenizer

    def __call__(self, text: str) -> float:
        X = build_X(sents=[text], tokenizer=self.tokenizer)
        H_last = self.rnn.forward(X)  # (N, L) -> (N, H)
        y_pred = self.rnn.W_hy(H_last)  # (N, H) * (?=H,?=1) -> (N, 1)
        # y_pred = 범위? =[-inf, +inf]
        # 그런데 우리가 원하는 건 확률.
        y_pred = torch.sigmoid(y_pred)
        return y_pred.item()


def main():
    # 데이터셋을 구축#
    sents = [
        sent
        for sent, _ in DATA
    ]
    labels = [
        label
        for _, label in DATA
    ]
    print(sents)
    print(labels)

    # 이제 뭐해요?
    # 경서님 - 토큰화 && 정수인코딩.
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts=sents)  # 정수인코딩 학습
    # seqs = tokenizer.texts_to_sequences(texts=sents)
    # for seq in seqs:
    #     print(seq)

    # 그럼 이제 뭐해요?
    # 정수인코딩의 나열의 길이가 다 다르다
    # 문제: 하나의 행렬 연산으로 다 계산할 수가 없다.
    # 패딩을 해서, 나열의 길이를 통일한다.
    # seqs = pad_sequences(sequences=seqs, padding="post", value=0)
    # for seq in seqs:
    #     print(seq)
    # 제일 길이가 긴 문장 = 16 = L.

    #  이제 뭐하죠?  패딩까지 마쳤다.
    #  정수의 나열.
    #  [난, 너, 를]  # categorical feature. class 의 개수 = 어휘의 크기.
    #  [1, 2, 3]
    #  단어 사이에도 크고 작음이 있네? 라고 학습할 것.
    # 그것을 바라지 않는다.
    # 1. one-hot 2. 임베딩 벡터.
    # e.g. [1, 2, 3] -> [[...], [...], [...]]
    #그런 벡터 표현을 학습해서 사용하는 것이 적절.
    # 경서님 - 중복되지 않은 토큰의 개수
    # = 어휘속에 있는 고유한 단어의 개수 = 학습하고자는 임베딩 벡터의 개수
    # = 어휘의 크기.
    vocab_size = len(tokenizer.word_index.keys())
    vocab_size += 1  # 왜 이걸 해줘야할까?
    # 경서님 - 0으로 패딩을 했기때문에, 패딩 토큰의 임베딩 벡터도 학습을 해야한다.
    # E = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=32)
    # # seqs: List[List[int]] (N, L).
    # Embeddings = E(torch.LongTensor(seqs))  # (N, L) -> (N, L, E).
    # print(Embeddings.shape)  # (42, 16, 32) = (42=데이터 개수,16=문장의 최대길이, 32=임베딩 벡터의 크기)

    # 이제 뭐해요?
    # 경서님 - 모델을 만들고 데이터를 나눠서 학습.
    # 임베딩 벡터는 학습되는 벡터이므로, 이것도 학습을 해야한다.
    # 임베딩 벡터는 어떻게 학습되는가?
    # 감성분석이라는 문제를 학습함과 동시에 임베딩 벡터를 학습하면된다.
    # 감성분석을 위한 loss.  -> 그 로스로 임베딩 벡터를 업데이트.
    # 그렇다면, 다음으로 필요한 작업은, 모델을 정의하고, 로스를 계산하는 로직을 정의.
    rnn = SimpleRNN(vocab_size, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE)
    optimizer = torch.optim.Adam(params=rnn.parameters(), lr=LR)

    # 이제 뭐해요?
    # 어떻게 학습?
    X: torch.Tensor = build_X(sents, tokenizer)
    y: torch.Tensor = build_y(labels)
    # loss = rnn.training_step(X, y)
    # loss.backward()  # 오차 역전파 끝. -> 편미분 값을 구할수 있게된다.
    # optimizer.step()  # 경사도 하강으로 W_hh, W_xh, W_hy가 최적화될것.
    # print(loss.item())

    # 이제 뭐해요?
    # X, y = 전체 데이터.
    # loss = 전체 데이터에 대해서 한번만 구한 것.

    for epoch in range(EPOCHS):
        loss = rnn.training_step(X, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  #  다음 에폭에서 기울기가 축적이 되지 않도록 리셋
        print(epoch, "-->", loss.item())





    # 이제 ㅜ머해요?
    # 로스 계산하면 끝?
    # 승환님 - 로스를 최소화.
    # 로스를 최소화는 어떻게?
    # 편미분. dLoss/dW_hh, dLoss/dW_xh, dLoss/dW_hy
    # W_hh = W_hh - LR * dLoss/dW_hh
    # W_xh = W_xh - LR * dLoss/dW_xh
    # W_hy = W_hy - LR * dLoss/dW_hy
    # 이렇게 경사도 하강을 하면, 웨이트가 최적화되고, 그러면 로스를 줄인다.

    # r그럼 편미분은 어떻게 구해여?
    # 역전파로 구해줘야 한다.

    # 이제 뭐하는가?
    # 학습된 모델로 테스트
    analyser = Anaylser(rnn, tokenizer)
    print("#### TEST ####")
    sent = "나는 자연어처리가 좋아"
    print(sent, "->", analyser(sent))


if __name__ == '__main__':
    main()
