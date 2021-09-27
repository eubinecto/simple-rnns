

from typing import List, Tuple
import numpy as np
import torch
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from torch.nn import functional as F


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.W_q = torch.nn.Linear(embed_size, hidden_size)
        self.W_k = torch.nn.Linear(embed_size, hidden_size)
        self.W_v = torch.nn.Linear(embed_size, hidden_size)
        # 새롭게 추가되는 가중치가 있나요?
        # concat된 피처를 다시 H로 조절해주는 선형변환
        self.W_o = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, X_embed: torch.Tensor) -> torch.Tensor:
        """
        :param X_embed: (N, L, E)
        :return: Out (N, L, H)
        """
        # 여기까지 X_embed이 왔다.
        # 이제 뭐하죠?
        # batched matrix multiplication
        # (N, L, E) * (E, H)
        # (E, H)를 N개 복사.
        # (N, L, E) * (N, E, H) ->  (N, L, H)
        Q = self.W_q(X_embed)  # (N, L, E) * (?=E, ?=H) -> (N, L, H)
        K = self.W_k(X_embed)  # (N, L, E) * (?=E, ?=H) -> (N, L, H)
        V = self.W_v(X_embed)  # (N, L, E) * (?=E, ?=H) -> (N, L, H)
        N, L, H = Q.size()

        Q = Q.reshape(N, self.num_heads, L, H // self.num_heads)  # (N, L, H) -> (N, heads, L, H / heads)
        K = K.reshape(N, self.num_heads, L, H // self.num_heads)  # (N, L, H) -> (N, heads, L, H / heads)
        V = V.reshape(N, self.num_heads, L, H // self.num_heads)  # (N, L, H) -> (N, heads, L, H / heads)
        # 이제 뭐하죠?
        # Q * K^T
        Q = Q / np.sqrt(self.hidden_size)
        K = K / np.sqrt(self.hidden_size)
        # Q * K^T
        # heads = e 로 표기.
        # H / heads = x로 표기.
        #  L, L = a, b로 표기.
        Out_ = torch.einsum("neax,nebx->neab", Q, K)  # (N, heads, L, H / heads) * (N, heads, L, H / heads)  -> (N, heads, L, L)
        # 이제 뭐하죠?
        # Q, K 의 H차원이 커지면 어떤 문제?
        # Out_ = Out_ / np.sqrt(self.hidden_size) -> 순서에 맞지는 않음. 먼저 Q와 K를 scale-down해줘야 한다.
        # 이제 뭐하죠?
        # 여기는 어떻게 바뀌어야 할까?
        # Out_ = torch.softmax(Out_, dim=1)  # across L  # (N, L, L)
        # Out_ = torch.softmax(Out_, dim=2)  # across L  # (N, L, L)
        Out_ = torch.softmax(Out_, dim=2)  # across L  # (N, heads,  L, L)
        # Out_ = torch.softmax(Out_, dim=3)  # across L  # (N, heads,  L, L) - 이것도 가능.
        # 이제 뭐하죠?
        # Out = torch.einsum("nll,nlh->nlh", Out_, V)  # (N,  L, L) * (N, L, H )-> (N, L, H)
        Out_ = torch.einsum("nell,nelx->nelx", Out_, V)  # (N, heads,  L, L) * (N, heads, L, H / heads)-> (N, heads, L, H / heads)
        Out_ = Out_.reshape(N, L, H)  # (N, heads, L, H / heads) -> (N, L, H)
        Out = self.W_o(Out_)  # (N, L, H) * (?=H,?=H) ->(N, L, H)
        # (H, H) -> N번 복사. -> (N, H, H)
        # (N, L, H) * (N, H, H) -> (..., L, H) * (..., H, H) ->  (N, L, H)
        # 끝인가요??
        return Out


class EncoderBlock(torch.nn.Module):

    def __init__(self, embed_size: int, hidden_size: int):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(embed_size, hidden_size)
        self.ffn = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, X_embed: torch.Tensor) -> torch.Tensor:
        """
        :param X_embed: (N, L, E)
        :return: (N, L, H)
        """
        # X_embed이 들어오면, 뭐해요?
        Out_ = self.self_attention(X_embed)  # (N, L, E) -> (N, L, H)
        # 다음에는 뭐해요?
        Out = self.ffn(Out_)  # (N, L, H) * (?=H, ?=H) -> (N, L, H)
        return Out


class Transformer(torch.nn.Module):
    # 1. 학습해야하는 가중치 (__init__)
    # 2. 신경망의 출력 (forward)

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, depth: int):
        """
        :param vocab_size: V
        :param embed_size: E
        :param hidden_size: H
        """
        super().__init__()
        # 학습해야하는 가중치를 정의합니다.
        self.embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.blocks = torch.nn.Sequential(
            # examples/explore_list_star 참고
            *[EncoderBlock(embed_size, hidden_size) for _ in range(depth)]
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L) 정수인코딩.
        :return:
        """
        X_embed = self.embeddings(X)  # (N, L) -> (N, L, E)
        # 이제는 뭘해야될까요?
        Out_ = self.blocks(X_embed)  # (N, L, E) -> (N, L, H)
        # 이제 뭘할까요?
        Out = ...
        return Out

    def training_step(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        Out = self.forward(X)
        # ...
        loss = ...
        return loss


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
    # transformer 으로 감성분석 문제를 풀기.
    transformer = Transformer(embed_size=EMBED_SIZE, vocab_size=vocab_size, num_heads=NUM_HEADS, depth=DEPTH, max_length=MAX_LENGTH)
    optimizer = torch.optim.Adam(params=transformer.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        loss = transformer.training_step(X, y)
        loss.backward()  # 오차 역전파
        optimizer.step()  # 경사도 하강
        optimizer.zero_grad()  # 다음 에폭에서 기울기가 축적이 되지 않도록 리셋
        print(epoch, "-->", loss.item())

    analyser = Analyser(transformer, tokenizer, MAX_LENGTH)
    print("##### TEST #####")
    for sent in TESTS:
        print(sent, "->", analyser(sent))


if __name__ == '__main__':
    main()
