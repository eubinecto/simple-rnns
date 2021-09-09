"""
감성분석기. charRNN으로 긍정/부정을 예측하는 모델.
anaylser("나는 자연어처리가 좋아") -> 0.9999
"""
from typing import List, Tuple
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import torch


DATA: List[Tuple[str, int]] = [
    # 긍정적인 문장 - 1
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
    ("나 보기가 역겨워", 0) # 긍정적인 확률이 0
]


# 신경망은 파이토치에서 정의하느냐  - Module 상속
class SimpleRNN(torch.nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_state: int):
        """
        h_t = f_W(h_t-1, xt)
        h_t = tanh(W_hh * h_t-1 + Wxh * xt)
        :param vocab_size:
        :param embed_size:
        """
        # hyper paramters
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_state = hidden_state
        super().__init__()
        # 학습해야하는 가중치들을 정의하는 곳.
        # 1. 감성분석에 적합한 의미를 가지는 임베딩 벡터를 어휘 속 모든 고유한 단어에 대하여 학습.
        self.E = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        # 2. 또 학습해야하는 가중치가 있나?
        # 가중치 행렬 둘.
        # 재완님 N(배치의 크기), L(가장 긴 문장의 길이), E(임베딩 벡터의 크기), H(hidden state의 크기)
        # 이중에 하나!
        # 그럼 이건 어떻게 결정?
        # 지금 결정을 하지 말고, 결정을 미루세요.
        # 데이터의 흐름을 보면서, 원하는 데이터의 차원이 나오도록 나중에 웨이트의 차원을 결정하면 된다.
        # -> 배치를 처리하는 로직을 봐야한다.
        self.W_hh = torch.nn.Linear(in_features=hidden_state, out_features=hidden_state)  # (?=H, ?=H)
        self.W_xh = torch.nn.Linear(in_features=embed_size, out_features=hidden_state)  # (?, ?)
        self.W_hy = torch.nn.Linear(in_features=hidden_state, out_features=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L) = N개의 최대길이 L인 정수인코딩의 나열
        :return:
        """
        # H_t = f_W(H_t-1, X_t)
        # 그럼 우선, t = 1, 2, 3, 4, 5 ,6.. 루프를 돌아야한다는 것을 알수 있습니다.
        # 재완님 - L  = 가장 긴 문장의 길이.
        H_t = torch.zeros(size=(X.shape[0], self.hidden_state))  # (N, H)
        for time in range(X.shape[1]):
            # (N, L) -> (N,)
            X_t = X[:, time]  # 배치 속 모든 데이터를 선택, 하지만 해당 시간대의 정수인코딩만을 선택
            X_t = self.E(X_t)  # (N,) -> (N, E)
            # self.W_hh(H_t) = (N, H) * (?=H, ?=H) = (N, H)
            # self.W_xh(X_t) = (N, E) * (?=E, ?=H) = (N, H)
            # 입력 행렬의 차원 * ??? = 출력 행렬의 차원
            # 그렇게 ???의 차원을 알아내면 편하다.
            H_t = torch.tanh(self.W_hh(H_t) + self.W_xh(X_t))
        # 이것을 목표로! 어떻게 가중치의 차원을 결정할 수 있을까?
        H_last: torch.Tensor = H_t  # (N, H)
        return H_last

    def training_step(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L)
        :param y: (N, 1)
        :return: loss (1)
        """
        H_last = self.forward(X)  # (N, L) -> (N, H)
        y_pred = self.W_hy(H_last)  # (N, H) * (?=H, ?=1) -> (N, 1)
        # Y 의 범위 = [-inf, +inf].
        # 우리는 이것을 [0, 1].
        y_pred = torch.sigmoid(y_pred)  # [-inf, +inf] -> [0, 1]
        # 목표 = 0.99, 0.87 ...  N개의 문장에 대하여 긍정적인 확률을 출력.
        # 그것과 010101 레이블을 비교하는 것.
        # y_pred (N, 1)
        # y (N, 1)
        # (N,1) 혹은 (N,)
        y_pred = torch.reshape(y_pred, y.shape)  # y와 y_pred의 차원을 동기화
        loss = F.binary_cross_entropy(input=y_pred, target=y)  # (N, 1), (N, 1) -> (N, 1)
        loss = loss.sum()  # 배치 = 로스를 계산하는 단위
        return loss


# builders 전체데이터 X, 전체데이터의 레이블 y를 텐서로 출력해주는 함수.
def build_X(sents: List[str], tokenizer: Tokenizer) -> torch.Tensor:
    seqs = tokenizer.texts_to_sequences(sents)
    seqs = pad_sequences(sequences=seqs, padding="post", value=0)
    return torch.LongTensor(seqs)


def build_y(labels: List[int]) -> torch.Tensor:
    return torch.FloatTensor(labels)


class SimpleDataset(Dataset):

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


class Anaylser:

    def __init__(self, rnn: SimpleRNN, tokenizer: Tokenizer):
        self.rnn = rnn
        self.tokenizer = tokenizer  # 새로운 데이터의 정수 인코딩.

    def __call__(self, text: str) -> float:
        """
        :param text: e.g. 나는 자연어처리가 좋아
        :return: 0.88
        """
        X = build_X(sents=[text], tokenizer=self.tokenizer)
        H_last = self.rnn.forward(X)  # (N, L) -> (N, H)
        y_pred = self.rnn.W_hy(H_last)  # (N, H) * (?=H, ?=1) -> (N, 1)
        # Y 의 범위 = [-inf, +inf].
        # 우리는 이것을 [0, 1].
        y_pred = torch.sigmoid(y_pred)  # [-inf, +inf] -> [0, 1]
        return y_pred.item()


# hyper paramters #
HIDDEN_SIZE = 64
EMBED_SIZE = 32
BATCH_SIZE = 30
LR = 0.001
EPOCHS = 200




def main():
    # 1. RNN의 입력으로 넣기 적합한 형태로 데이터 전처리
    sents = [
        sent
        for sent, label in DATA
    ]
    print("문장의 개수", len(sents))
    labels = [
        label
        for sent, label in DATA
    ]
    # print(sents)
    # print(labels)
    # 이제 뭐하지?
    # 예은님 - 학습? - 학습을 진행하기 전에, 데이터 전처리.
    # 토큰화 안녕 -> 안, 녕. 정수인코딩.
    tokenizer = Tokenizer(char_level=True)
    # 우리의 문장을 학습
    tokenizer.fit_on_texts(texts=sents)  # 정수인코딩을 학습.
    # seqs = tokenizer.texts_to_sequences(texts=sents)
    #
    # for seq in seqs:
    #     print(seq)
    #
    # # 정수인코딩도 끝났다. 그럼 이제 뭐해요?
    # # 정수인코딩된 나열의 길이가 다 다르다.
    # # -> 하나의 행렬연산으로 계산이 불가능.
    # # 선홍님 - 0으로 채워넣는다 - padding. 패딩을 해줘야한다.
    # seqs = pad_sequences(sequences=seqs, padding='post', value=0)
    # for seq in seqs:
    #     print(seq)

    # 패딩까지 끝
    # 그럼 이제 뭐해요? = 그럼 이제 문제가 남아있죠?
    # [난, 너, 를] = categorical features/
    # [1, 2, 3] = 대소관계.
    # 글자 사이에도 대소관계가 있다라는 것을 학습해버릴수도 있다.
    # 1. 원핫 벡터 (어휘의 크기가 너무 크지 않다면) 2. 임베딩 벡터.
    # 학습할 임베딩의 개수 = ???
    # 성민님 - 어휘속에 존재하는 고유한 단어의 개수만큼 임베딩 벡터를 학습.
    # 어휘의 크기.
    vocab_size: int = len(tokenizer.word_index.keys())
    vocab_size += 1  # 이건 왜? 왜 고유한 단어의 개수가 하나더?
    # 선홍님 : 새로운 토큰 0 (패딩 토큰)
    # # 이 토큰의 임베딩 벡터도 학습을 해야한다. 아마도 "별 의미없는 토큰"이라는 것을 학습.
    # E = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBED_SIZE)
    #
    # X = E(torch.LongTensor(seqs))  # 정수인코딩의 나열 -> 임베딩 벡터의 나열
    # print(X.shape)  # (38, 16, 32) = (N, L, E)
    # # [[1, 2, 3],
    #  [4, 5, 6]]
    # [[...], [...], [...],
    #  [...], [...], [...]]
    # N = 2, E = 32, L = 3(가장 긴 문장의 길이)

    # 임베딩 벡터를 구하는 것까지 끝.
    # 이제 뭐해요?
    # 민우님 - 학습. - 임베딩 벡터를 학습해야한다.
    # 임베딩 벡터를 학습하기위해선 그럼 뭘해야하죠?
    # anaylser("나는 자연어처리가 좋아") -> 0.9999
    # 이런 감성분석 문제를 학습함으로서 임베딩 벡터도 동시에 학습을 하면 된다.
    # -> 모델을 정의를 하고, 모델 학습진행.

    X = build_X(sents, tokenizer)
    y = build_y(labels)
    dataset = SimpleDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


    rnn = SimpleRNN(vocab_size, EMBED_SIZE, HIDDEN_SIZE)

    # 경사도 하강의 일종
    # params = 최적화 해야하는 파라미터를 지정
    optimizer = torch.optim.Adam(params=rnn.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        losses = list()
        for batch in loader:
            X, y = batch
            # 해당 배치의 로스 계산.
            loss = rnn.training_step(X, y)  # (N, L), (N, 1) -> (1).
            # 이제 뭐해요?
            # 역전파. 목표 = dLoss/dW. 이것을 chain rule로 다 계산.
            loss.backward()  # 오차역전파 끝. (기울기 계산).
            # 그다음엔 뭐해요? 기울기 계산한뒤에는?
            # 재왼님 - weight 갱신 ; weight = weight - LR * dLoss/weight
            optimizer.step()  # 경사도 하강으로 가중치 갱신.
            optimizer.zero_grad()  # 기울기가 축적이 되지 않도록 이전 기울기를 리셋.
            losses.append(loss.item())
        avg_loss = (sum(losses)/len(losses))
        print(epoch, "->", avg_loss)


    print("###### test #####")
    analyser = Anaylser(rnn, tokenizer)
    sent = "나는 자연어처리가 좋아"
    print(sent, "->", analyser(sent))
    sent = "도움이 되었으면 좋겠습니다"
    print(sent, "->", analyser(sent))


if __name__ == '__main__':
    main()
