import numpy as np
import torch
from typing import List, Tuple
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from torch.nn import functional as F


class MultiHeadSelfAttentionLayer(torch.nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        assert hidden_size % heads == 0
        self.W_q = torch.nn.Linear(embed_size, hidden_size)
        self.W_k = torch.nn.Linear(embed_size, hidden_size)
        self.W_v = torch.nn.Linear(embed_size, hidden_size)
        # multihead attention의 경우 - 학습해야하는 가중치가 더 있나요?
        self.W_o = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, X_embed: torch.Tensor) -> torch.Tensor:
        """
        :param X_embed: (N, L, E)
        :return: H_attn (N, L, H)
        """
        # 셀프 어텐션에서는 각 세개의 신경망에 들어가는 입력이 동일하다.
        Q = self.W_q(X_embed)  # (N, L, E) * (?=E,?=H) -> (N, L, H)
        K = self.W_k(X_embed)  # (N, L, E) * (?=E,?=H) -> (N, L, H)
        V = self.W_v(X_embed)  # (N, L, E) * (?=E,?=H) -> (N, L, H)
        # 이제 뭘해야할까...
        # H를 heads가 사이좋게 나눠쓰도록 하면 된다.
        N, L, H = Q.size()
        # 새로운 차원을 추가해야한다. - heads 차원.
        Q = Q.reshape(N, self.heads, L, H // self.heads)  # (N, L, H) -> (N, heads, L, H / heads)
        K = K.reshape(N, self.heads, L, H // self.heads)  # (N, L, H) -> (N, heads, L, H / heads)
        V = V.reshape(N, self.heads, L, H // self.heads)  # (N, L, H) -> (N, heads, L, H / heads)
        # 이 다음엔 뭘하죠?
        # Q와 K의 내적.
        Q /= np.sqrt(self.hidden_size)
        K /= np.sqrt(self.hidden_size)
        # 영성님 - 왜 각각 적용시켜야 하는건가요?
        # Q ~ (0, 1)
        # K ~ (0, 1)
        # Q * K^T ~ (0, d_k)
        # Q * K^T  scale -> ~ (0, 1)
        # 그렇다면,
        # Q scale -> Q ~ (0,  1 / d_k)
        # K scale -> K ~ (0,  1 / d_k)
        # Q * K^T ~ (0, 1)
        # 단어와 단어사이의 관계. (N, L, H), (N, L, H) -> (N, L, H) * (N, H, L) -> (N, L, L)
        # Out_ = torch.einsum("nah,nbh->nab", Q, K)
        # heads = e
        # H / heads = x
        Out_ = torch.einsum("neax,nebx->neab", Q, K)  # (..., ..., L, H / heads), (... ..., L, H / heads) -> (..., ..., L, H / heads) * (... ..., H / heads, L) -> ( ..., ..., L ,L)
        # Out_ = Out_ / np.sqrt(self.hidden_size)  # 매우 크거나 매우 작은수가 나올 확률을 줄여주는 것. - 소 잃고 외양간 고치는 오류.
        # Out_ = torch.softmax(Out_, dim=1)  # [0, 1]사이로  정규화   # (N, L, L)
        Out_ = torch.softmax(Out_, dim=2)  # [0, 1]사이로  정규화  # (N, heads, L, L)
        # Out = torch.einsum("nll,nlh->nlh", Out_, V)  # Out(N, L, L), V(N, L, H)  -> (N, L, H)
        Out_ = torch.einsum("nell,nelx->nelx", Out_, V)  # (..., ..., L, L) * (..., ..., L, H / heads) -> (..., ...,  L, H / heads)
        # 이제 뭘하죠?
        # concat
        # Out = torch.cat(Out_[:, :], dim=1)
        Out_ = Out_.reshape(N, L, self.hidden_size)  # (N, heads,  L, H / heads) -> (N, L, H)
        # 끝?
        Out = self.W_o(Out_)  # (N, L, H) * (?=H, ?=H) -> (N, L, H)
        return Out


class EncoderBlock(torch.nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, heads: int):
        super().__init__()
        self.embed_size = embed_size
        # 학습해야하는 가중치?
        # 영성님 - self_attention
        # position-wise FFN
        self.sa = MultiHeadSelfAttentionLayer(embed_size=embed_size, hidden_size=hidden_size, heads=heads)
        self.norm_1 = torch.nn.LayerNorm(hidden_size)
        self.ffn = torch.nn.Linear(in_features=embed_size, out_features=embed_size)
        self.norm_2 = torch.nn.LayerNorm(hidden_size)

    def forward(self, X_embed: torch.Tensor) -> torch.Tensor:
        """
        :param X_embed: (N, L, E)
        :return: (N, L, E)
        """
        Out_ = self.sa(X_embed) + X_embed  # (N, L, E) -> (N, L, E)
        Out_ = self.norm_1(Out_)
        # L, E * E, E -> L, E.
        Out_ = self.ffn(Out_) + Out_  # (N, L, E) * (?=E,?=E) -> (N, L, E)
        Out = self.norm_2(Out_)
        return Out


class Transformer(torch.nn.Module):
    # 1. 신경망 속에 학습해야하는 가중치를 정의하는 메소드.
    # 2. 신경망의 출력을 계산하는 메소드.
    # 3. 신경망의 로스를 계산하는 메소드.
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, depth: int, max_length: int, heads: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        # --- 가중치를 정의하는 곳 --- #
        # 임베딩 테이블 정의
        self.token_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.pos_embeddings = torch.nn.Embedding(num_embeddings=max_length, embedding_dim=embed_size)
        # 인코더 레이어 정의
        self.blocks = torch.nn.Sequential(
            *[EncoderBlock(embed_size, hidden_size, heads) for _ in range(depth)]
        )
        self.W_hy = torch.nn.Linear(hidden_size, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L)
        :return:
        """
        # 정수인코딩 -> 임베딩.
        X_embed = self.token_embeddings(X)  # (N, L) -> (N, L, E)
        # torch.nn.Sequential로 정의를 했기 때문에, blocks를 레이어 취급할 수 있다.
        # [1 , 512, 5, 43]
        # [0,   1,  2,  3]
        # [[...], [...], [...], [...]]
        X_pos = self.pos_embeddings(torch.arange(self.max_length).expand(X.shape[0], self.max_length))  # (N, L)  -> (N, L, E)
        H_all = self.blocks(X_embed + X_pos)  # (N, L, E) -> (N, L, E)
        return H_all

    def predict(self, H_all: torch.Tensor) -> torch.Tensor:
        H_avg = H_all.mean(dim=1)  # (N, L,H) -> (N, H)
        y_pred = self.W_hy(H_avg)  # (N, H) * (?, ?) -> (N, 1)
        return torch.sigmoid(y_pred)

    def training_step(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        H_all = self.forward(X)  # (N, L) -> (N, L, H)
        y_pred = self.predict(H_all)  # (N, L, H) -> (N,)
        y_pred = y_pred.reshape(y.shape)
        loss = F.binary_cross_entropy(y_pred, y)
        return loss.sum()




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
        H_all = self.transformer.forward(X)
        y_pred = self.transformer.predict(H_all)
        return y_pred.item()


EPOCHS = 100
HIDDEN_SIZE = 512
EMBED_SIZE = 512
MAX_LENGTH = 30
NUM_HEADS = 8
DEPTH = 3
LR = 0.0001



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
    transformer = Transformer(embed_size=EMBED_SIZE,  hidden_size=HIDDEN_SIZE, vocab_size=vocab_size,
                       heads=NUM_HEADS, depth=DEPTH, max_length=MAX_LENGTH)
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