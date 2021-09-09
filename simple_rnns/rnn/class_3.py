"""
감성분석을 할 수 있는  charRNN을 pytorch로 밑바닥부터 구현하는 것.
ananylser("난 너무 기분이 좋아") -> 0.80
"""
from typing import List, Tuple
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import torch


# pytorch로 신경망을 구축하기 위해선, 클래스를 정의하고, torch.nn.Module을 상속하면 된다.
from torch.utils.data.dataset import T_co


class SimpleRNN(torch.nn.Module):
    """
    감성분석 - many to one.
    one - h_last.
    """
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        """
        h_t = f_w(h_t-1, x_t)
        h_t = tanh( W_hh* h_t-1  + W_xh * x_t)
        h_t의 차원의 크기 = hidden_size
        """
        super().__init__()
        # 하이퍼 파라미터 저장
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        # 신경망의 레이어들을 정의한다.
        self.embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        # x* W+ b
        # 먼저 X (N, L)
        # embedding layer 를 거치면
        # X -> embeddings (N, L) -> (N, E)
        # X * W_xh = (N, E) * ???(E, H) = (N, H)
        # H_t-1:  (N, H)
        # H_t-1 * W_hh = (N, H) * ??? (H, H) = (N, H)
        self.W_xh = torch.nn.Linear(in_features=embed_size, out_features=hidden_size)
        self.W_hh = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        #  N개의 문장 별로, 하나의 값 (긍정/부정)을 출력하고 싶은 것.
        # (N, H) * ??? (H, 1) =  (N, 1)
        self.W_hy = torch.nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: 입력 (N, L)
        :return: 또다른 텐서 (forward pass). h_last.
        h_t = f_w(h_t-1, x_t)
        h_t = tanh( W_hh* h_t-1  + W_xh * x_t)
        """
        H_last = torch.zeros(size=(X.shape[0], self.hidden_size))  # (N, H)
        X = self.embeddings(X)  # (N, L) -> (N, E)
        # X_1 = X[:, 0]
        # H_1 = torch.tanh(self.W_hh(H_0) + self.W_xh(X_1))
        # X_2 = X[:, 1]
        # H_2 = torch.tanh(self.W_hh(H_1) + self.W_xh(X_2))
        # X_3 = X[:, 2]
        # H_3 = torch.tanh(self.W_hh(H_2) + self.W_xh(X_3))
        # t: 0 -> L- 1
        # 문장의 길이가 길어지면 학습이 느려질 것. O(n).
        for time in range(X.shape[1]):
            # X[: , t] - 모든 배치를 선택, 하지만 time 시간대의 임베딩 벡터만 선택.
            # W_hh(H_last) - (N, H) * (H, H) -> (N, H)
            # W_xh(X[: , time]) - (N, L, E) -> (N, E) -> (N, E) * (E, H) -> (N, H)
            # (N, H) + (N, H)
            H_last = torch.tanh(self.W_hh(H_last) + self.W_xh(X[:, time]))
        return H_last

    def training_step(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param X: 배치의 입력
        :param y: 배치의 정답 [1, 0, 1, 0 0]
        :return:
        """
        H_last = self.forward(X)  # (N, L) -> (N, H).
        #  N개의 문장 별로, 하나의 값 (긍정/부정)을 출력하고 싶은 것.
        Hy = self.W_hy(H_last)  # (N, H) * ??? (H, 1) -> (N, 1)
        # Hy의 범위? - [0, 1] 가 아니고, [-inf, +inf]
        Hy = torch.sigmoid(Hy)  # (N,  1) -> (N , 1)
        # 벡터를 1차원 행렬로 표현할지 (N, 1), 아니면 그냥 벡터로 표현할지 (N,)를 결정해야하는데,
        # y의 차원을 따라감으로서 통일한다.
        Hy = torch.reshape(Hy, y.shape)
        #  배치는 로스를 계산하는 단위.
        # (N, 1)/(N,) -> 1
        loss = F.binary_cross_entropy(input=Hy, target=y).sum()
        return loss


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
    ("그렇게 살아서 뭘하겠니", 0)
]

# --- hyper parameters --- #
EMBED_SIZE = 10
HIDDEN_SIZE = 32
BATCH_SIZE = 13
EPOCHS = 300
LR = 0.001


class SimpleDataset(Dataset):

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        # X, y 전체 데이터를 저장.
        self.X = X
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        a = A()
        a[0]
        dataset = Dataset()
        dataset[0], dataset[1]
        :param index:
        :return:
        """
        return self.X[index], self.y[index]


#  X 와 y를 구축하기 위한 함수를 꼭 정의합니다 (저는요)
def build_X(sents: List[str], tokenizer: Tokenizer) -> torch.Tensor:
    seqs = tokenizer.texts_to_sequences(texts=sents)  # 정수 인코딩
    seqs = pad_sequences(sequences=seqs, padding="post", value=0)  # 패딩
    return torch.LongTensor(seqs)  # 텐서로 출력


def build_y(labels: List[int]) -> torch.Tensor:
    return torch.FloatTensor(labels)


# 학습이 완료된 RNN을 활용하여, 감성분석을 하는 클래스를 정의.

class SimpleSentimentAnalyser:

    def __init__(self, rnn: torch.nn.Module, tokenizer: Tokenizer):
        self.rnn = rnn
        self.tokenizer = tokenizer

    def __call__(self, text: str) -> float:
        """
        난 널 사랑해 -> 0.9999
        :param text:
        :return:
        """
        X = build_X(sents=[text], tokenizer=self.tokenizer)
        H_last = self.rnn.forward(X) #  (N, L) -> (N, H)
        Hy = self.rnn.W_hy(H_last)  # (N, H) * ??? (H, 1) -> (N, 1)
        # Hy의 범위? - [0, 1] 가 아니고, [-inf, +inf]
        Hy = torch.sigmoid(Hy)  # (N,  1) -> (N , 1)
        return Hy.item()


def main():
    sents = [
        sent
        for sent, _ in DATA
    ]
    labels = [
        label
        for _, label in DATA
    ]

    # 이제 뭘해야되지?
    # 다운님 -토크나이징
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts=sents)   # 문장을 글자단위로 토크나이즈해주고, 동시에 정수인코딩도 학습.
    # seqs = tokenizer.texts_to_sequences(texts=sents)
    # for seq in seqs:
    #     print(seq)
    #
    # # 이제 뭘해야하지? - 길이를 패딩을 통해서 통일한다.
    # # padding token 의 정수 인코딩 = 0.
    # seqs = pad_sequences(sequences=seqs, padding="post", value=0)
    # for seq in seqs:
    #     print(seq)
    # # 이제 뭐해요? 정훈님 - 임베딩
    # # 테이블의 row의 개수.
    # # 정훈님 - 제일 큰 정수값
    # # 임베딩의 개수 = 고유한 단어의 개수.
    # # 앰베딩으 개수 = 어휘의 크기.
    vocab_size = len(tokenizer.word_index.keys())
    print(0 in tokenizer.word_index.values())  # 정수 인코딩은 0부터 시작하는 것이 아니라, 1부터 시작한다.
    # 왜 하나 더 추가?
    # 패딩을 할 때, 0을 패딩 토큰으로 사용했기 때문에, 고유한 토큰이 하나더 늘어난다.
    vocab_size += 1
    rnn = SimpleRNN(vocab_size=vocab_size, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE)
    rnn.train()  # 내부의 가중치가 변경되도록 허용
    # 전체 데이터셋을 구축  #
    X: torch.Tensor = build_X(sents, tokenizer)
    y: torch.Tensor = build_y(labels)
    dataset = SimpleDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)

    #
    # for batch in loader:
    #     print(batch)

    # 데이터 셋 구축 완료.
    # 학습 진행.
    # 배치 단위로 로스를 계산.
    for epoch in range(EPOCHS):
        losses = list()
        for b_idx, batch in enumerate(loader):
            X, y = batch
            loss = rnn.training_step(X, y)
            loss.backward()   # 오차역전파.
            # 최종 로스에 대하여, 모든 가중치에 대한 편미분을 구했다.
            # 경사도 하강으로 업데이트.
            optimizer.step()  # 가중치 최적화 진행
            optimizer.zero_grad()  # 기울기가 축적되지 않도록 역전파된 기울기를 리셋.
            losses.append(loss.item())
        avg_loss = (sum(losses) / len(losses))
        print("epoch={}, avg_loss={}".format(epoch, avg_loss))


    print("-----training test------")
    analyser = SimpleSentimentAnalyser(rnn, tokenizer)
    for sent, label in DATA:
        print(sent, label, analyser(sent))

    print("--- 학습하지않은 문장을 넣어보기 --- ")
    print(analyser("짐승이랑 다를게 없네"))


if __name__ == '__main__':
    main()
