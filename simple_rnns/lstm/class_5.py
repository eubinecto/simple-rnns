"""
목표  = 감성분석기, LSTM.

"""
from typing import List, Tuple
import torch
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from torch.nn import functional as F


DATA: List[Tuple[str, int]] = [
    # 긍정적인 문장 - 1
    ("나는 자연어처리가 좋아", 1),
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
    ("너는 어렵다", 0),
    # ("자연어처리는", 0)
]


# builders #
def build_X(sents: List[str], tokenizer: Tokenizer, max_length: int) -> torch.Tensor:
    seqs = tokenizer.texts_to_sequences(texts=sents)
    seqs = pad_sequences(sequences=seqs, padding="post", maxlen=max_length, value=0)
    return torch.LongTensor(seqs)


def build_y(labels: List[int]) -> torch.Tensor:
    return torch.FloatTensor(labels)


# --- hyper parameters --- #
EPOCHS = 800
LR = 0.001
HIDDEN_SIZE = 32
EMBED_SIZE = 12
MAX_LENGTH = 30


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
        # 학습해야하는 가중치.
        # 학습해야하는 신경망을 다 정의해보자.
        # 1. 정의를 하는 것 스킵
        # 2. forward에서 데이터의 흐름을 *차원의 변화를 트래킹*하면서 확인
        # 3. (A, B) * (B, C) -> (A, C)
        # self.W_f = torch.nn.Linear(in_features=hidden_size + embed_size, out_features=hidden_size)  # (?, ?)
        # self.W_i = torch.nn.Linear(in_features=hidden_size + embed_size, out_features=hidden_size)  # (?, ?)
        # self.W_o = torch.nn.Linear(in_features=hidden_size + embed_size, out_features=hidden_size)  # (?, ?)
        # self.W_h_temp = torch.nn.Linear(in_features=hidden_size + embed_size, out_features=hidden_size)  # (?, ?)
        self.W = torch.nn.Linear(in_features=hidden_size + embed_size, out_features=hidden_size*4)
        self.W_hy = torch.nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        h_t = f_W(h_t-1, xt)
        :param X: (N, L) = (배치의 크기, 문장의 길이)
        :return: H_last (N, H)
        """
        # TODO:
        ...
        # 이제 뭐해요?
        # 반복문
        # 0 -> ???=L-1
        C_t = torch.zeros(size=(X.shape[0], self.hidden_size))  # (?=N, ?=H)
        H_t = torch.zeros(size=(X.shape[0], self.hidden_size))  # (?=N, ?=H)
        for time in range(X.shape[1]):
            # 여기선 뭐하죠?
            X_t = X[:, time]  # (N, L) -> (N, 1)
            X_t = self.E(X_t)  # (N, 1) -> (N, E)
            # 이제 뭐하죠?
            H_cat_X = torch.cat([H_t, X_t], dim=1)  # (N, H), (N, E) -> (?=N,?=H+E)
            # 성분곱, 성분덧셈만을 해서 H_t를 구해야하므로, F,O, I, G는전부 (N, H)
            # F = self.W_f(H_cat_X)  # (N, H +E) * (?=H+E, ?=H) = (?=N, ?=H)
            # I = self.W_i(H_cat_X)  # (N, H +E) * (?=H+E, ?=H) = (?=N, ?=H)
            # O = self.W_o(H_cat_X)  # (N, H +E) * (?=H+E, ?=H) = (?=N, ?=H)
            # H_temp = self.W_h_temp(H_cat_X)  # (N, H +E) * (?=H+E, ?=H) = (?=N, ?=H)
            G = self.W(H_cat_X)  # (N, H+E) * (H+E, H*4) -> (N, H*4)
            F = G[:, :self.hidden_size]  # 큰 박스 안에 작은 박스를 넣는다.
            I = G[:, self.hidden_size:self.hidden_size*2]  # 큰 박스 안에 작은 박스를 넣는다.
            O = G[:, self.hidden_size*2:self.hidden_size*3]  # 큰 박스 안에 작은 박스를 넣는다.
            H_temp = G[:, self.hidden_size*3:self.hidden_size*4]  # 큰 박스 안에 작은 박스를 넣는다.

            # 이제 뭐하죠? - 활성화 함수,
            F = torch.sigmoid(F)  #  (N ,H) -> (N, H)
            I = torch.sigmoid(I)  #  (N ,H) -> (N, H)
            O = torch.sigmoid(O)  #  (N ,H) -> (N, H)
            H_temp = torch.tanh(H_temp)  #  (N ,H) -> (N, H)

            # 이제 뭐하죠?
            # 1. F의 용도? = 지우개
            # 2. I의 용도? = 필요한 기억 저장
            # 3. O의 용도? = 장기기억으로 부터 최종 단기기억 생성
            C_t = torch.mul(F, C_t) + torch.mul(I, H_temp)
            H_t = torch.mul(O, torch.tanh(C_t))
        # 마지막 시간대의 hidden state.
        H_last: torch.Tensor = H_t
        return H_last

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        H_last = self.forward(X)  # (N, L) -> (N, H)
        y_pred = self.W_hy(H_last)  # (N, H) -> (N, 1)
        y_pred = torch.sigmoid(y_pred)  # (N, H) -> (N, 1)
        # print(y_pred)
        return y_pred

    def training_step(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred = self.predict(X)  # (N, L) -> (N, 1)
        y_pred = torch.reshape(y_pred, y.shape)  # y와 차원의 크기를 동기화
        print(y_pred)
        loss = F.binary_cross_entropy(y_pred, y)  # 이진분류 로스
        loss = loss.sum()  # 배치 속 모든 데이터 샘플에 대한 로스를 하나로
        return loss


class Analyser:
    """
    lstm기반 감성분석기.
    """
    def __init__(self, lstm: SimpleLSTM, tokenizer: Tokenizer, max_length: int):
        self.lstm = lstm
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, text: str) -> float:
        X = build_X(sents=[text], tokenizer=self.tokenizer, max_length=self.max_length)
        y_pred = self.lstm.predict(X)
        return y_pred.item()


def main():
    # 문장을 다 가져온다
    sents = [sent for sent, label in DATA]
    # 레이블
    labels = [label for sent, label in DATA]
    # 정수인코딩
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts=sents)
    # 데이터 셋을 구축. (X=(N, L), y=(N, 1))
    X = build_X(sents, tokenizer, MAX_LENGTH)
    y = build_y(labels)

    vocab_size = len(tokenizer.word_index.keys())
    vocab_size += 1
    # lstm으로 감성분석 문제를 풀기.
    lstm = SimpleLSTM(vocab_size=vocab_size,hidden_size=HIDDEN_SIZE, embed_size=EMBED_SIZE)
    optimizer = torch.optim.Adam(params=lstm.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        loss = lstm.training_step(X, y)
        loss.backward()  # 오차 역전파
        optimizer.step()  # 경사도 하강
        optimizer.zero_grad()  #  다음 에폭에서 기울기가 축적이 되지 않도록 리셋
        print(epoch, "-->", loss.item())


    analyser = Analyser(lstm, tokenizer, MAX_LENGTH)
    print("##### TEST #####")
    sent = "나는 자연어처리가 좋아"
    print(sent, "->", analyser(sent))

if __name__ == '__main__':
    main()
