"""
LSTM을 파이토치로 구현해보기!
"""

from typing import List, Tuple
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import torch
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


TEST = [
    ("행복한 쉬는날", 1),
    ("운수좋은 날", 1),
    ("도저히 못 해 먹겠다", 0)
]

# builders #
def build_X(sents: List[str], tokenizer: Tokenizer, max_length: int) -> torch.Tensor:
    seqs = tokenizer.texts_to_sequences(texts=sents)
    seqs = pad_sequences(sequences=seqs, padding="post", maxlen=max_length, value=0)
    return torch.LongTensor(seqs)


def build_y(labels: List[int]) -> torch.Tensor:
    return torch.FloatTensor(labels)


class SimpleLSTM(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, embed_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        # 여기는 학습해야하는 가중치를 정의하는 부분.
        # 우리의 목표는, 이 가중치 행렬을 최적화 하는 것.
        # 이 가중치의 shape는 어떻게 결정?
        # 1. 일단 정의만 하고 스킵
        # 2. 데이터의 흐름을 차원을 트래킹하면서 확인
        # 3. (A, B) * (B, C) = (A, C)
        self.E = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.W = torch.nn.Linear(in_features=self.hidden_size + self.embed_size, out_features=self.hidden_size*4)
        self.W_hy = torch.nn.Linear(in_features=hidden_size, out_features=1)  # (H, 1)
        # 웨이트의 초기값을 더 적절하게 설정
        torch.nn.init.xavier_uniform_(self.E.weight)
        torch.nn.init.xavier_uniform_(self.W.weight)
        torch.nn.init.xavier_uniform_(self.W_hy.weight)

    def forward(self, X: torch.Tensor):
        """
        :param X: (N, L)
        :return:
        """
        # 영성님 - torch.zeroes
        C_t = torch.zeros(size=(X.shape[0], self.hidden_size))  # (N, H)
        H_t = torch.zeros(size=(X.shape[0], self.hidden_size))  # (N, H)
        for time in range(X.shape[1]):
            X_t = X[:, time]  # (N, L) -> (N, 1)
            X_t = self.E(X_t)  # (N, 1) -> (N, E)
            H_cat_X = torch.cat([H_t, X_t], dim=1)  # (N, H), (N, E) -> (N, H+E)
            Gates = self.W(H_cat_X)  # (N, H+E) * (?=H+E, ?=H*4) -> (N, H * 4)
            I = torch.sigmoid(Gates[:, :self.hidden_size])# 입력게이트의 출력값.
            F = torch.sigmoid(Gates[:, self.hidden_size:self.hidden_size*2])  # 망각게이트의 출력값.
            O = torch.sigmoid(Gates[:, self.hidden_size*2:self.hidden_size*3]) # 출력게이트의 출력값.
            H_temp = torch.tanh(Gates[:, self.hidden_size*3:self.hidden_size*4])  # 임시 단기기억. (G)/
            # 이제 뭐해요?
            # 영성님 현재 시간대 장기기억 C_t를 만들어봅시다!
            # 뭘잊어야 하나? torch.mul(F, C_t)  (N, H) 성분곱 (N, H) -> (N, H)
            C_t = torch.mul(F, C_t) + torch.mul(I, H_temp)
            # O (N, H) -> 범위 = [0, 1]
            # 필요없는 기억에 해당하는 차원 = 0에 가깝게 알아서 조절
            # 필요한 기억의 차원 = 1에 가깝게 알아서 조절.
            H_t = torch.mul(O, torch.tanh(C_t))  # (N, H) 성분곱 (N,H) -> (N, H)
        return H_t

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        H_t = self.forward(X)  # (N, L) -> (N, H)
        y_pred = self.W_hy(H_t)  # (N, H) -> (N, 1)
        y_pred = torch.sigmoid(y_pred)  # (N, H) -> (N, 1)
        return y_pred

    def training_step(self, X: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        # 배치가 들어왔을 때, 로스를 계산하면 된다.
        H_last = self.forward(X)  # (N, L) -> (N, H)
        # 이것으로부터, 긍정 / 부정일 확률을 구하고 싶다.
        # (N, H) * ???-(H, 1) -> (N, 1) - 어떻게 할 수 있을까?
        Hy = self.W_hy(H_last)  # (N, H) * (H, 1) -> (N, 1)
        # 그럼 Hy 의 범위가  [0, 1]?  아니다.
        # (N, 1)
        Hy_normalized = torch.sigmoid(Hy)  # [-inf, inf] -> [0, 1]
        # (N, 1), (N,) -> 하나로 통일하기 위해, y의 맞춘다.
        Hy_normalized = torch.reshape(Hy_normalized, y.shape)
        # print(Hy_normalized)
        # N개의 로스를 다 더해서, 이 배치의 로스를 구한다.
        loss = F.binary_cross_entropy(input=Hy_normalized, target=y).sum()
        return loss



# hyper paraamters #
EPOCHS = 900
LR = 0.0001
HIDDEN_SIZE = 32
EMBED_SIZE = 12
MAX_LENGTH = 30


class Analyser:

    def __init__(self, lstm: SimpleLSTM, tokenizer: Tokenizer, max_length: int):
        self.lstm = lstm
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, text: str) -> float:
        X = build_X(sents=[text], tokenizer=self.tokenizer, max_length=self.max_length)
        y_pred = self.lstm.predict(X)
        return y_pred.item()


def main():
    # 문장들
    sents = [sent for sent, label in DATA]
    # 레이블들
    labels = [label for sent, label in DATA]
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts=sents)
    X: torch.Tensor = build_X(sents, tokenizer, MAX_LENGTH)  # (N, L)
    y: torch.Tensor = build_y(labels)  # (N, 1)

    vocab_size = len(tokenizer.word_index.keys())
    vocab_size += 1

    lstm = SimpleLSTM(vocab_size, HIDDEN_SIZE, EMBED_SIZE)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        loss = lstm.training_step(X, y)  # loss 계산.
        loss.backward()  # 오차 역전파
        optimizer.step()  # 경사도 하강
        optimizer.zero_grad()  #  다음 에폭에서 기울기가 축적이 되지 않도록 리셋
        print(epoch, "-->", loss.item())

    analyser = Analyser(lstm, tokenizer, MAX_LENGTH)

    print("#### TRAIN TEST ####")
    # traing accuracy
    correct = 0
    total = len(DATA)
    for sent, y in DATA:
        y_pred = analyser(sent)
        if y_pred >= 0.5 and y == 1:
            correct += 1
        elif y_pred < 0.5 and y == 0:
            correct += 1
    print("training acc:", correct / total)


    print("#### TEST ####")
    for sent, y in TEST:
        print(sent, y, analyser(sent))


if __name__ == '__main__':
    main()
