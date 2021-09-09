
"""
LSTM을 활용한 감성분석.
"""
from typing import List, Tuple
import torch
from torch.nn import functional as F
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

DATA: List[Tuple[str, int]] = [
    # 긍정적인 문장 - 1
    # ("나는 자연어처리", 1),
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
    ("너는 어렵다", 0),
    ("자연어처리는", 0)
]

TEST_DATA = [
    ("자연어처리는 재밌어", 1),
    ("자연어처리는 어렵다", 0),
    ("이해가 안되네요", 0)
]

class SimpleLSTM(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, embed_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        # 학습해야하는 가중치가 무엇 무엇이 있었나?
        # 입력.. 출력을 어떻게 해야하나...?
        # 우선, 학습해야하는 가중치는 이 둘이다.
        self.E = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        # 임베딩 레이어를 학습해야한다.
        # 학습해야하는 가중치를 정의하는 곳.
        # 질문 - 또 학습해야하는 가중치?
        # 이 가중치의 shape를 어떻게 결정?
        # 1. 가중치를 정의하고, shape를 결정하는 건 스킵.
        # 2. 데이터의 흐름을 **차원을 트래킹**하면서 체크.
        # 3. (A, B) * (B, C) -> (A, C).
        self.W = torch.nn.Linear(in_features=self.hidden_size + self.embed_size, out_features=self.hidden_size*4)  # (?=H+E, ?=H*4)
        self.W_hy = torch.nn.Linear(in_features=hidden_size, out_features=1)
        torch.nn.init.xavier_uniform_(self.E.weight)
        torch.nn.init.xavier_uniform_(self.W.weight)
        torch.nn.init.xavier_uniform_(self.W_hy.weight)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L)
        :return:
        """
        # 이제 뭐해요?
        # 다운님
        # (N, L) * (?=L, ?=H) -> (N, H) 이렇게 한방에 되지는 않을 것.
        # 이전시간대의 장기기억
        C_t = torch.zeros(size=(X.shape[0], self.hidden_size))  # (N, H)
        # 이전시간대의 단기기억
        H_t = torch.zeros(size=(X.shape[0], self.hidden_size))  # (N, H)
        for time in range(X.shape[1]):
            pass
            # 이제 뭐해요?  -> 이전 시간대의 장기 & 단기기억 준비완료.
            # 이제 X_t가 필요.
            X_t = X[:, time]  # (N, L) -> (N, 1)
            # 이제 뭐해요?
            X_t = self.E(X_t)  # (N, 1) -> (N, E)
            # 정훈님 - f, i, o, 임시단기기억.
            H_cat_X = torch.cat([H_t, X_t], dim=1)  # (N, H), (N, E) -> (?=N, ?=H+E)
            G = self.W(H_cat_X)  # (N, H+E) * (?=H+E, ?=H*4) -> (N, H*4).
            F = torch.sigmoid(G[:, :self.hidden_size])  # (N, H*4)  -> (N, H)
            I = torch.sigmoid(G[:, self.hidden_size:self.hidden_size*2])
            O = torch.sigmoid(G[:, self.hidden_size*2:self.hidden_size*3])
            H_temp = torch.tanh(G[:, self.hidden_size*3:self.hidden_size*4])
            # 현시간대의 장기기억?
            C_t = torch.mul(F, C_t) + torch.mul(O, H_temp)  # 성분곱이니, 차원은 변하지 않는다 (N,H)
            H_t = torch.mul(O, torch.tanh(C_t))  # -> (N,H)
        H_last: torch.Tensor = H_t  # (N, H)
        return H_last

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        H_t = self.forward(X)  # (N, L) -> (N, H)
        y_pred = self.W_hy(H_t)  # (N, H) -> (N, 1)
        y_pred = torch.sigmoid(y_pred)  # (N, H) -> (N, 1)
        # print(y_pred)
        return y_pred


    def training_step(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred = self.predict(X)  # (N, L) -> (N, 1)
        y_pred = torch.reshape(y_pred, y.shape)  # y와 차원의 크기를 동기화
        loss = F.binary_cross_entropy(y_pred, y)  # 이진분류 로스
        loss = loss.sum()  # 배치 속 모든 데이터 샘플에 대한 로스를 하나로
        return loss


# builders #
def build_X(sents: List[str], tokenizer: Tokenizer, max_length: int) -> torch.Tensor:
    seqs = tokenizer.texts_to_sequences(texts=sents)
    seqs = pad_sequences(sequences=seqs, padding="post", maxlen=max_length, value=0)
    return torch.LongTensor(seqs)


def build_y(labels: List[int]) -> torch.Tensor:
    return torch.FloatTensor(labels)


EPOCHS = 600
LR = 0.001
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
    sents = [sent for sent, label in DATA]
    labels = [label for sent, label in DATA]
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts=sents)
    X = build_X(sents, tokenizer, MAX_LENGTH)
    y = build_y(labels)

    vocab_size = len(tokenizer.word_index.keys())
    vocab_size += 1
    lstm = SimpleLSTM(vocab_size=vocab_size,hidden_size=HIDDEN_SIZE, embed_size=EMBED_SIZE)
    optimizer = torch.optim.Adam(params=lstm.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        loss = lstm.training_step(X, y)
        loss.backward()  # 오차 역전파
        optimizer.step()  # 경사도 하강
        optimizer.zero_grad()  #  다음 에폭에서 기울기가 축적이 되지 않도록 리셋
        print(epoch, "-->", loss.item())

    analyser =Analyser(lstm, tokenizer, MAX_LENGTH)

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
    for sent, label in TEST_DATA:
        print(sent, label, analyser(sent))


if __name__ == '__main__':
    main()
