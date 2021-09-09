"""
감성분석을 할수 있는 charRNN모델을 만들 것.
anaylser("난 자연어처리가 좋아") -> 0.99999
"""

from typing import List, Tuple
import torch.nn
from keras_preprocessing.text import Tokenizer
from torch.nn import functional as F
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co

DATA: List[Tuple[str, int]] = [
    # 긍정적인 문장 - 1
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
    ("배고파 죽을 것 같아", 0)
]


# 심층신경망 - torch.nn.Module
class SimpleRNN(torch.nn.Module):

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        """
        h_t = f_W(h_t-1, x_t)
        h_t = tanh(W_hh * h_t-1 + W_xh * x_t)
        """
        #  hidden_size = 뉴런의 개수.
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        super().__init__()
        # 학습해야하는 가중치를 정의하는 구간.
        # 학습하애하는 가중치는 어떻게 정의?
        # 그런데 이 shape를 어떻게 찾을 수 있을까?
        self.embedding_layer = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.W_hh = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size) # (H, H)
        self.W_xh = torch.nn.Linear(in_features=embed_size, out_features=hidden_size)  # (E, H)
        self.W_hy = torch.nn.Linear(in_features=hidden_size, out_features=1)

    # 데이터의 흐름을 보자
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L)=  (데이터의 개수, 문장의 최대길이)
        :return:
        """
        X = self.embedding_layer(X)  # (N, L) -> (N, L, E).
        # X_1 = X[:, 0, :]  # 배치 속에 있는 모든 데이터를 선택 하되, 시간대가 0인 것만 선택. # (N, L, E) -> (N, 1, E) =  (N, E)
        # X_2 = X[:, 1, :]
        # X_3 = X[:, 2, :]
        # H_0 = torch.zeros(size=(X.shape[0], self.hidden_size))  # (N, H)
        # # self.W_hh(H_0) :(N, H) * ???(H ,H)  -> (N, H)
        # # self.W_xh(X_1) :(N, E) * ???(E, H) -> (N, H)
        # H_1 = torch.tanh(self.W_hh(H_0) + self.W_xh(X_1))  # (N, H) + (N, H)  --> (N, H)
        # H_2 = torch.tanh(self.W_hh(H_1) + self.W_xh(X_2))  # (N, H) + (N, H)  --> (N, H)
        # H_3 = torch.tanh(self.W_hh(H_2) + self.W_xh(X_3))  # (N, H) + (N, H)  --> (N, H)
        #
        # range는 무엇을 넣어야하나?
        # 윤서님 - 어떤것의 행의 수?
        # 왜 0으로 채워지나?
        # H_0
        H_last = torch.zeros(size=(X.shape[0], self.hidden_size))
        # 문장의 길이가 길어지면, 느릴수 밖에 없겠다. forward - O(N).
        for time in range(X.shape[1]):
            X_t = X[:, time]
            H_last = torch.tanh(self.W_hh(H_last) + self.W_xh(X_t))
        return H_last

    def training_step(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        배치가 주어졌을 때, 로스를 계산.
        :param X: (N, L)
        :param y: (N, 1) e.g. 0101010101
        :return: (1,)
        """
        H_last = self.forward(X)   # (N, L) -> (N, H).
        # (N, H) * ???(H, 1) -> (N, 1)
        Hy = self.W_hy(H_last)  # (N, H) -> (N, 1). 치역? [-inf, +inf]
        # [-inf, +inf] -> [0, 1]
        Hy = torch.sigmoid(Hy)  # (N, 1) -> (N, 1) 치역 [0, 1]
        # (N, 1) 혹은 (N,)
        Hy = torch.reshape(Hy, y.shape)  # y와 Hy의 차원 동기화
        loss = F.binary_cross_entropy(input=Hy, target=y)  # (N, 1) , (N, 1) -> (N, 1)
        loss: torch.Tensor = loss.sum()  # (N,1) -> (1,)
        return loss


class SimpleDataset(Dataset):

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


# builders
def build_X(sents: List[str], tokenizer: Tokenizer) -> torch.Tensor:
    seqs = tokenizer.texts_to_sequences(sents)
    seqs = pad_sequences(sequences=seqs, padding="post", value=0)
    return torch.LongTensor(seqs)


def build_y(labels: List[int]) -> torch.Tensor:
    return torch.FloatTensor(labels)



class Analyser:

    def __init__(self, rnn: SimpleRNN, tokenizer: Tokenizer):
        self.rnn = rnn
        self.tokenizer = tokenizer

    def __call__(self, text: str) -> float:
        X = build_X(sents=[text], tokenizer=self.tokenizer)
        H_last = self.rnn.forward(X)
        Hy = self.rnn.W_hy(H_last)
        Hy = torch.sigmoid(Hy)
        return Hy.item()


EMBED_SIZE = 12
BATCH_SIZE = 30
HIDDEN_SIZE = 64 # 뉴런의 개수
LR = 0.001
EPOCHS = 300


def main():
    # --- 데이터 구축 ---- #
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

    # 이제 뭘해야하지?
    # 모델이 글자를 이해하지는 않는다.
    # 모델의 입력은 숫자가 들어가야한다.
    # 토큰화 & 정수인코딩 (e.g. 원핫인코딩)
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts=sents)  # 정수인코딩을 학습하게 된다.

    seqs = tokenizer.texts_to_sequences(sents)

    for seq in seqs:
        print(seq)

    # 다음엔 뭘해야하지?
    # 준우님 - 길이 통일.
    # padding을 한다는 것.
    seqs = pad_sequences(sequences=seqs, padding="post", value=0)
    for seq in seqs:
        print(seq)

    # padding 까지 끝.
    # 이제 뭐해야 하나요?
    #  0, 1 있는 곳에.. 토큰화한 것을 대입?  - labels 은 확률입니당
    #  정규화라기보단. 벡터화. 각 단어를 벡터화하는 과정
    # 1. 원핫-인코딩, 임베딩.
    # 학습할 임베딩 벡터의 개수?
    vocab_size = len(tokenizer.word_index.keys())
    # 왜 하나더? 이거 안하면 오류가 뜬다.
    vocab_size += 1  # padding 토큰의 임베딩 벡터도 학습을 해야한다.
    # embedding_layer = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBED_SIZE)
    # # 임베딩 벡터
    # embeddings = embedding_layer(torch.LongTensor(seqs))
    # print(embeddings.shape)  # 2 차원? 3차원? (32, 15, 12) = (N, L, E)
    # ['난', '널, '사']
    # [ 1, 2 , 3]
    # [[[...], [...] , [...]],
    #  [[...], [...] , [...]]]   - > (2 , 3, 12)

    # 이제 뭐해야하지?
    # 0과 1을 다시데려온다.
    # (input, target) 페어 구축?
    # 1. Dataset, DataLoader
    X = build_X(sents, tokenizer)
    y = build_y(labels)
    dataset: Dataset = SimpleDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    rnn = SimpleRNN(vocab_size, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE)
    optimizer = torch.optim.Adam(params=rnn.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        losses = list()
        for batch in loader:
            # (입력, 출력)페어.
            X, y = batch
            loss = rnn.training_step(X, y)   # (N, L) , (N, 1) -> (1,)
            # 배치에 대응하는 로스를 계산했다. 이제 뭘해야되나?
            loss.backward()  # 오차역전파. 끝. 최종 로스의 각 가중치에 대한 편미분 값.
            # 기울기를 구했다.
            optimizer.step()  # 경사도 하강으로 가중치 업데이트
            optimizer.zero_grad()  # 이전에 구했던 기울기를 축적하지 않기 위해, 역전파된 기울기를 리셋.
            losses.append(loss.item())
        avg_loss = (sum(losses) / len(losses))
        print(epoch, "->", avg_loss)

    analyser = Analyser(rnn, tokenizer)

    for sent, label in DATA:
        print(sent, label, analyser(sent))

    print("나는 자연어처리가 좋아", analyser("나는 자연어처리가 좋아"))


if __name__ == '__main__':
    main()