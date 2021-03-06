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
        # 병렬화를 위해, 모든 self-attention
        assert hidden_size % num_heads == 0  # embed_size must be divisible by num_heads.
        self.W_q = torch.nn.Linear(in_features=embed_size, out_features=hidden_size, bias=False)
        self.W_k = torch.nn.Linear(in_features=embed_size, out_features=hidden_size, bias=False)
        self.W_v = torch.nn.Linear(in_features=embed_size, out_features=hidden_size, bias=False)
        self.W_o = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L, E)
        :return: H_all (N, L, E)
        """
        Q = self.W_q(X)  # (N, L, E) * (E, E) -> (N, L, H)
        K = self.W_k(X)  # (N, L, E) * (E, E) -> (N, L, H)
        V = self.W_v(X)  # (N, L, E) * (E, E) -> (N, L, H)
        # 헤드로 나누기 - 가중치의 역할 = transform into heads.
        N, L, H = Q.size()
        Q = Q.reshape(N, self.num_heads, L, H // self.num_heads)  # (N, heads, L,  E / heads)
        K = K.reshape(N, self.num_heads, L, H // self.num_heads)  # (N, heads, L,  E / heads)
        V = V.reshape(N, self.num_heads, L, H // self.num_heads)  # (N, heads, L,  E / heads)
        # scaled
        Q = Q / np.sqrt(self.hidden_size)
        K = K / np.sqrt(self.hidden_size)
        # batched matrix multiplication.
        H_all_ = torch.einsum("nhax,nhbx->nhab", Q, K) # (N, heads, L,  H / heads), (N,  heads, L, H / heads)  -> (N, heads, L, L)
        H_all_ = torch.softmax(H_all_, dim=2)
        H_all_ = torch.einsum("nhll,nhlx->nhlx", H_all_, V)  # (N, heads, L, L), (N, head , L, H / heads) -> (N, heads, L, H / heads)
        H_all_ = H_all_.reshape(N, L, H)  # (N, heads, L, H / heads)  -> (N, L, H)
        H_all = self.W_o(H_all_)  # (N, L, H) * (H, H) -> (N, L, H)
        return H_all


class EncoderLayer(torch.nn.Module):

    def __init__(self, embed_size: int, hidden_size: int, num_heads: int, max_length: int):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_length = max_length
        # --- layers --- #
        self.mhsa = MultiHeadSelfAttention(embed_size, hidden_size, num_heads)
        self.norm_1 = torch.nn.LayerNorm(embed_size)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(embed_size, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, embed_size)
        )
        self.norm_2 = torch.nn.LayerNorm(embed_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L, E)
        :return: H_all (N, L, E)
        """
        H_all_ = self.mhsa(X) + X  # (N, L, E) -> (N, L, E) & residual connection
        H_all_ = self.norm_1(H_all_)
        H_all_ = self.ffn(H_all_) + H_all_  # (N, L, E) -> (N, L, E) & residual connection
        H_all = self.norm_2(H_all_)
        return H_all


class Transformer(torch.nn.Module):
    """
    Transformer for classifying sequences
    """
    def __init__(self, embed_size,  hidden_size, vocab_size, num_heads, depth, max_length):
        """
        :param embed_size: Embedding dimension
        :param hidden_size:
        :param vocab_size: Number of tokens (usually words) in the vocabulary
        :param num_heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param max_length: Expected maximum sequence length
        """
        super().__init__()
        # --- hyper parameters --- #
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        # --- embeddings --- #
        self.token_embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.pos_embedding = torch.nn.Embedding(num_embeddings=max_length, embedding_dim=embed_size)
        # --- encoder blocks --- #
        self.blocks = torch.nn.Sequential(*[
                EncoderLayer(embed_size, hidden_size, num_heads, max_length)
                for _ in range(depth)
        ])
        # --- the last layer --- #
        self.W_hy = torch.nn.Linear(embed_size, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L) A batch by sequence length integer tensor of token indices.
        :return: the binary classification probability.
        """
        X_E = self.token_embedding(X)  # (N, L) -> (N, L, E)
        X_pos = torch.arange(X.shape[1]).expand(X.shape[0], X.shape[1])  # (N, L) -> (N, L)
        X_pos = self.pos_embedding(X_pos)  # (N, L) -> (N, L, E)
        X = X_E + X_pos  # (N, L, E) + (N, L, E) -> (N, L, E) - component-wise addition.
        H_all = self.blocks(X)  # (N, L, E) -> (N, L, E)
        H_avg = H_all.mean(dim=1)
        y_pred = self.W_hy(H_avg)
        y_pred = torch.sigmoid(y_pred)
        return y_pred

    def training_step(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred = self.forward(X)
        y_pred = y_pred.reshape(y.shape)
        loss = F.binary_cross_entropy(y_pred, y)
        return loss.sum()


# --- hyper parameters --- #
EPOCHS = 100
EMBED_SIZE = 512
HIDDEN_SIZE = 512
MAX_LENGTH = 30
NUM_HEADS = 8
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
    lstm = Transformer(embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, vocab_size=vocab_size, num_heads=NUM_HEADS, depth=DEPTH, max_length=MAX_LENGTH)
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
