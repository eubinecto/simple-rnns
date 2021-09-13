"""
목표 = seq2seq.
with AttentionLSTM
"""
from typing import List, Tuple
import torch
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from konlpy.tag import Okt
from torch.nn import functional as F
import termplotlib as tpl


DATA: List[Tuple[str, str]] = [
    # 긍정적인 문장 - 1
    ("나는 너를 좋아해", "I love you"),
]


# builders #
def build_X_M(sents: List[str], tokenizer: Tokenizer, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    for building X and the masks.
    :param sents:
    :param tokenizer:
    :param max_length:
    :return:
    """
    seqs = tokenizer.texts_to_sequences(texts=sents)
    seqs = pad_sequences(sequences=seqs, padding="post", maxlen=max_length, value=0)
    X = torch.LongTensor(seqs)  # (N, L)
    M = torch.where(X != 0, 1, X)   # 마스크. padding = 0, else = 1   # (N, L)
    return X, M  # (N, L), (N, L)


def build_y(labels: List[int]) -> torch.Tensor:
    return torch.FloatTensor(labels)


class SimpleLSTM(torch.nn.Module):
    """
    LSTM,
    """
    def __init__(self, vocab_size: int, hidden_size: int, embed_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        # 여기에 학습할 가중치들을 정의해두어야.. module.parameters()를 했을 때, 불러 올 수 있다.
        self.E = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.W = torch.nn.Linear(in_features=hidden_size + embed_size, out_features=hidden_size*4)

    def forward(self, X: torch.Tensor, H_0: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h_t = f_W(h_t-1, xt)
        :param X: (N, L) = (배치의 크기, 문장의 길이)
        :param H_0: (N, H).
        :return: States (N, H*L)
        """
        # TODO:
        ...
        # 이제 뭐해요?
        # 반복문
        # 0 -> ???=L-1
        C_t = torch.zeros(size=(X.shape[0], self.hidden_size))  # (?=N, ?=H)
        H_t = H_0 if H_0 else torch.zeros(size=(X.shape[0], self.hidden_size))  # (?=N, ?=H)
        H_all = torch.zeros(size=(X.shape[0], X.shape[1], self.hidden_size))  # (N, L, H).
        for time in range(X.shape[1]):
            # 여기선 뭐하죠?
            X_t = X[:, time]  # (N, L) -> (N, 1)
            X_t = self.E(X_t)  # (N, 1) -> (N, E)
            # 이제 뭐하죠?
            H_cat_X = torch.cat([H_t, X_t], dim=1)  # (N, H), (N, E) -> (?=N,?=H+E)
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
            H_all[:, time] = H_t
        H_last = H_t
        return H_all, H_last  # (N, L, H), (N, H).

    # def predict(self, X: torch.Tensor, M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     :param X: (N, L) inputs.
    #     :param M: (N, L) masks.
    #     """
    #     # we need .. attention mask.
    #     H_all, H_last = self.forward(X)  # (N, L) -> (N, L, H),  (N, H).
    #     S = torch.einsum("nlh,nh->nl", H_all, H_last)  # (N, L, H), (N, H) -> (N, L)
    #     # mask the scores - we must set this to be negative.
    #     S[M == 0] = float("-inf")  # masking must be done.
    #     A = torch.softmax(S, dim=1)  # attention scores.
    #     C = torch.einsum("nlh,nl->nh", H_all, A)  # (N, L, H), (N, L) -> (N, H).
    #     y_pred = self.W_hy(torch.cat([C, H_last], dim=1))  # (N, H*2) @ *(H*2, 1) -> (N, 1)
    #     y_pred = torch.sigmoid(y_pred)  # (N, H) -> (N, 1)
    #     # print(y_pred)
    #     return y_pred, A

    # def training_step(self, X: torch.Tensor, M: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    #     y_pred, _ = self.predict(X, M)  # (N, L) -> (N, 1)
    #     y_pred = torch.reshape(y_pred, y.shape)  # y와 차원의 크기를 동기화
    #     loss = F.binary_cross_entropy(y_pred, y)  # 이진분류 로스
    #     loss = loss.sum()  # 배치 속 모든 데이터 샘플에 대한 로스를 하나로
    #     return loss


class SimpleSeq2Seq(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, embed_size:int):
        super().__init__()
        # we've got an encoder and a decoder!
        self.encoder = SimpleLSTM(vocab_size, hidden_size, embed_size)
        self.decoder = SimpleLSTM(vocab_size, hidden_size, embed_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X:
        :return:
        """
        pass

    def training_step(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        returns the loss.
        :param X:
        :param y:
        :return:
        """
        pass


# --- hyper parameters --- #
EPOCHS = 800
LR = 0.001
HIDDEN_SIZE = 32
EMBED_SIZE = 12
MAX_LENGTH = 30
PAD_TOKEN = "[PAD]"
PAD_IDX = 0


def main():
    # 문장을 다 가져온다
    sents = [sent for sent, label in DATA]
    # 레이블
    labels = [label for sent, label in DATA]
    # 정수인코딩
    tokenizer = Tokenizer(char_level=True, )
    tokenizer.fit_on_texts(texts=sents)
    # 데이터 셋을 구축. (X=(N, L, 2), y=(N, 1))
    X, M = build_X_M(sents, tokenizer, MAX_LENGTH)
    y = build_y(labels)
    vocab_size = len(tokenizer.word_index.keys())
    vocab_size += 1
    # seq2seq로 번역문제 풀기!
    model = ...
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        loss = model.training_step(X, M, y)
        loss.backward()  # 오차 역전파
        optimizer.step()  # 경사도 하강
        optimizer.zero_grad()  #  다음 에폭에서 기울기가 축적이 되지 않도록 리셋
        print(epoch, "-->", loss.item())

    # analyser = Analyser(lstm, tokenizer, MAX_LENGTH)
    # print("##### TRAIN TEST #####")
    # for sent, label in DATA:
    #     print(sent, "->", label, analyser(sent))
    # print("##### TEST #####")
    # sent = "나는 자연어처리가 좋아"
    # print(sent, "->", analyser(sent))


if __name__ == '__main__':
    main()
