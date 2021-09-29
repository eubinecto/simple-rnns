


from typing import List, Tuple
import numpy as np
import torch
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from torch.nn import functional as F


class MultiHeadSelfAttentionLayer(torch.nn.Module):

    def __init__(self, embed_size: int, hidden_size: int, heads: int):
        super().__init__()
        self.heads = heads
        self.hidden_size = hidden_size
        assert hidden_size % heads == 0
        self.W_q = torch.nn.Linear(embed_size, hidden_size)
        self.W_k = torch.nn.Linear(embed_size, hidden_size)
        self.W_v = torch.nn.Linear(embed_size, hidden_size)
        # 학습해야하는 가중치?
        self.W_o = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, X_embed: torch.Tensor) -> torch.Tensor:
        Q = self.W_q(X_embed)  #  (N, L, E) * (?=E, ?=H) -> (N, L, H)
        K = self.W_k(X_embed)  #  (N, L, E) * (?=E, ?=H) -> (N, L, H)
        V = self.W_v(X_embed)  #  (N, L, E) * (?=E, ?=H) -> (N, L, H)
        # 이제 뭐해야하나요?
        # 현재, hidden vector가 N (배치의 크기) * L (문장의 길이) 개 존재.
        # multihead attention -> hidden vector 의 개수 = (N,  heads, L,  H / heads)
        N, L, H = Q.size()
        # 성분의 개수가 동일하므로, 차원의 개수를 늘려서 재배치가 가능.
        Q = Q.reshape(N, self.heads, L, H // self.heads)  # (N, L, H) ->  (N,  heads, L,  H / heads)
        K = K.reshape(N, self.heads, L, H // self.heads)  # (N, L, H) ->  (N,  heads, L,  H / heads)
        V = V.reshape(N, self.heads, L, H // self.heads)  # (N, L, H) ->  (N,  heads, L,  H / heads)

        # 다음 - Q * K^T. (N, L, H), (N, L, H) -> (N, L, L)
        # Out = torch.einsum("nah,nbh->nab", Q, K)
        # heads = e
        # H / heads = x
        # 유사도 행렬을 구하는 부분.
        Q = Q / np.sqrt(self.hidden_size)
        K = K / np.sqrt(self.hidden_size)
        Out_ = torch.einsum("neax,nebx->neab", Q, K)  # (N,  heads, L,  H / heads) * (N,  heads, L,  H / heads) -> (N ,heads, ?=L, ?=L)
        # scale by sqrt H
        # Out_ = Out_ / np.sqrt(self.hidden_size)  # 왜 실수? 논리적인 오류. - 소 잃고 외양간 고치는 격.
        Out_ = torch.softmax(Out_, dim=2)  #  (N ,heads, ?=L, ?=L)
        Out_ = torch.einsum("neab,neax->neax", Out_, V)  # (N ,heads, ?=L, ?=L), (N,  heads, L,  H / heads) -> (N,  heads, L,  H / heads)
        # Out_ = torch.softmax(Out_, dim=3)  # (N ,heads, ?=L, ?=L)
        # Out_ = torch.einsum("neab,nebx->nebx", Out_, V)  # (N ,heads, ?=L, ?=L), (N,  heads, L,  H / heads) -> (N,  heads, L,  H / heads)

        # Out_ = torch.softmax(Out_, dim=3)  #  (N ,heads, ?=L, ?=L) - 이것도 맞음.
        # softmax() * V
        Out_ = Out_.reshape(N, L, H)  # (N,  heads, L,  H / heads) ->  (N, L , H / heads * heads=H)
        # "jointly learning"
        Out = self.W_o(Out_)  # (N, L, H) * (?=H, ?=H) -> (N, L, H)
        return Out


class EncoderBlock(torch.nn.Module):

    def __init__(self, embed_size: int, hidden_size: int, heads: int):
        super().__init__()
        self.sa = MultiHeadSelfAttentionLayer(embed_size, hidden_size, heads)
        self.layer_norm_1 = torch.nn.LayerNorm(hidden_size)
        self.ffn = torch.nn.Linear(hidden_size, hidden_size)
        self.layer_norm_2 = torch.nn.LayerNorm(hidden_size)

    def forward(self, X_embed: torch.Tensor) -> torch.Tensor:
        """
        :param X_embed: (N, L, E)
        :return:
        """
        # 1. self-attention을 통과했을 떄, 어떤 차원의 행렬이 나와야하나?
        Out_ = self.sa(X_embed) + X_embed  # (N, L, E) -> (N, L, H)
        Out_ = self.layer_norm_1(Out_)  # 특정 평균 or 분산에 치우친 분포를 이상적인 분포로 scale
        Out = self.ffn(Out_) + Out_  # (N, L, H) * (???) - (N, L, H)
        Out_ = self.layer_norm_2(Out_)
        return Out


class Transformer(torch.nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, vocab_size: int, embed_size: int,  hidden_size: int, heads: int, depth: int, max_length: int):
        """
        :param vocab_size: V
        :param embed_size: E
        """
        super().__init__()
        # 가중치를 정의하는 곳.
        self.max_length = max_length
        self.token_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        # 위치 정보의 경우 - 몇개의 임베딩을 학습? - 문장의 최대길이 만큼.
        self.pos_embeddings = torch.nn.Embedding(num_embeddings=max_length, embedding_dim=embed_size)  # 일단 스킵.
        # 여러 개 레이어의 나열을 하나의 레이어로 정의하는 방법
        # list of torch.nn.Module
        self.blocks = torch.nn.Sequential(
            *[EncoderBlock(embed_size, hidden_size, heads) for _ in range(depth)]
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X:  (N, L)
        :return: (N, L, H)
        """
        X_embed = self.token_embeddings(X)  # (N, L) -> (N, L, E)
        # torch.arange()
        # max_length = L.
        X_pos = self.pos_embeddings(torch.arange(self.max_length).expand(X.shape[0], self.max_length))  # (N, L) -> (N, L, ?)
        Input = X_embed + X_pos  # (N, L, E) + (N, L, ?=E)
        Out = self.blocks(Input)  # (N, L, E) -> (N, L, H)
        return Out

    def training_step(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass


# --- hyper parameters --- #
EPOCHS = 50
EMBED_SIZE = 512
MAX_LENGTH = 30
NUM_HEADS = 10
DEPTH = 3
LR = 0.0001


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
    ("나는 너가 싫어", 0),
    ("잘도 그러겠다", 0),
    ("너는 어렵다", 0),
    # ("자연어처리는", 0)
]

TESTS = [
    "나는 자연어처리가 좋아",
    "나는 자연어처리가 싫어",
    "나는 너가 좋다",
    "너는 참 좋다",
]


# builders #
def build_X(sents: List[str], tokenizer: Tokenizer, max_length: int) -> torch.Tensor:
    seqs = tokenizer.texts_to_sequences(texts=sents)
    seqs = pad_sequences(sequences=seqs, padding="post", maxlen=max_length, value=0)
    return torch.LongTensor(seqs)


def build_y(labels: List[int]) -> torch.Tensor:
    return torch.FloatTensor(labels)


class Analyser:
    """
    lstm기반 감성분석기.
    """
    def __init__(self, transformer: Transformer, tokenizer: Tokenizer, max_length: int):
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, text: str) -> float:
        X = build_X(sents=[text], tokenizer=self.tokenizer, max_length=self.max_length)
        y_pred = self.transformer.forward(X)
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
    # lstm 으로 감성분석 문제를 풀기.
    lstm = Transformer(embed_size=EMBED_SIZE, vocab_size=vocab_size, num_heads=NUM_HEADS, depth=DEPTH, max_length=MAX_LENGTH)
    optimizer = torch.optim.Adam(params=lstm.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        loss = lstm.training_step(X, y)
        loss.backward()  # 오차 역전파
        optimizer.step()  # 경사도 하강
        optimizer.zero_grad()  # 다음 에폭에서 기울기가 축적이 되지 않도록 리셋
        print(epoch, "-->", loss.item())

    analyser = Analyser(lstm, tokenizer, MAX_LENGTH)
    print("##### TEST #####")
    for sent in TESTS:
        print(sent, "->", analyser(sent))


if __name__ == '__main__':
    main()
