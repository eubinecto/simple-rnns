"""
감성분석을 할 수 있는 charRNN 모델을 구현하는 것.
e.g.
analyser("난 너가 좋아") -> 0.999
analyser("난 너가 싫어") -> 0.01
"""
from typing import List, Tuple
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from torch.nn import functional as F
import torch
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
    ("주식이 떨어졌다", 0)
]


class SimpleDataset(Dataset):

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        # X - y 페어 저장.
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]




# X 와 y를 구축하는 함수 두개
def build_X(sents: List[str], tokenizer: Tokenizer) -> torch.Tensor:
    seqs = tokenizer.texts_to_sequences(texts=sents)
    # 다음엔 뭘하지? - 길이를 통일하기 위해 패딩을 진행
    seqs = pad_sequences(sequences=seqs, padding="post", value=0)
    # 정수텐서로 출력
    return torch.LongTensor(seqs)


def build_y(labels: List[int]) -> torch.Tensor:
    # 레이블은 확률이기 때문에, float으로 출력.
    return torch.FloatTensor(labels)


# 신경망 정의 = Module 상속
class SimpleRNN(torch.nn.Module):
    """
    감성분석 모델 - many to one  모델.

    """

    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int):
        """
        h_t = f_W(h_t-1, x_t)
        h_t = tanh(h_t-1 * W_hh +  x_t * W_xh)
        """
        super().__init__()
        # hyper parameters
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        #  학습이 필요한 가중치를 선언해주는 부분.
        # 학습해야하는 가중치 행렬.
        # x * W + b
        # 데이터가 들어왔을 때 차원이 어떻게 바뀔까?
        # X (N, L)
        # embedding layer: (N, L) -> (N, L, E).
        # X (N, L, E).
        self.embedding_layer = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.W_hh = torch.nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.W_xh = torch.nn.Linear(in_features=self.embed_size, out_features=self.hidden_size)
        self.W_hy = torch.nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L) e.g. [[1,2,3,0,0], [1,1,2,3, 0]]
        :return:
        h_t = f_W(h_t-1, x_t)
        h_t = tanh(h_t-1 * W_hh +  x_t * W_xh)
        """
        #  모델의 출력을 계산하여 출력.
        X = self.embedding_layer(X)  # (N, L) -> (N, L, E)
        # H_0 = torch.zeros(size=(X.shape[0], self.hidden_size))  # (N, H).
        # # # H_1 = tanh(H_0 * W_hh + X_1 * W_xh)
        # # X_1 = X[:, 0]  # (N, L, E) -> (N, E)
        # # # H_1: (N, H)
        # # # self.W_hh(H_0)  (N, H) * ???(H,H) = (N, H)
        # # # self.W_xh(X_1) (N, E) * ???(E,H) = (N, H)
        # # H_1 = torch.tanh(self.W_hh(H_0) + self.W_xh(X_1))
        # # X_2 = X[:, 1]  # (N, L, E) -> (N, E)
        # # H_2 = torch.tanh(self.W_hh(H_1) + self.W_xh(X_2))
        # # X_3 = X[:, 2]  # (N, L, E) -> (N, E)
        # # H_3 = torch.tanh(self.W_hh(H_2) + self.W_xh(X_3))
        H_last = torch.zeros(size=(X.shape[0], self.hidden_size))
        # X = (N, L, E)
        # 학습하는데 걸리는 시간이 나열의 길이에 비례.
        # O(n).
        # 병렬화가 불가능. 재귀적으로 정의.
        for time in range(X.shape[1]):
            X_t = X[:, time]
            H_last = torch.tanh(self.W_hh(H_last) + self.W_xh(X_t))
        return H_last

    def training_step(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, L)
        :param y: (N,) e.g. [0,1,1,0,1]
        :return:
        """
        #  하나의 배치에 대한 로스를 계산.
        # 마지막 단기기억.
        H_last = self.forward(X)  # (N, L) -> (N, H)
        # 어떻게 마지막 단기기억을 바탕으로 이 문장의 긍정/부정 확률을 출력할 수 있을까?
        # (N, H) * W_hy (H, 1) -> (N, 1)
        Hy = self.W_hy(H_last)  # (N, H) -> (N, 1) - 범위 = (-inf, +inf)
        Hy = torch.sigmoid(Hy)  # (N, 1) -> (N, 1) - 범위 (0, 1)
        # N개의 데이터에 대한 긍정일 확률이 출력.
        # (N, 1), (N,)
        Hy = torch.reshape(Hy, y.shape)
        loss = F.binary_cross_entropy(input=Hy, target=y)  # (N, 1), (N, 1) -> (N, 1)
        # 해당 배치에 대한 로스결과 출력.
        loss: torch.Tensor = loss.sum()  # (N, 1) -> 1
        return loss



# hyper parameters
EMBED_SIZE = 50
BATCH_SIZE = 26
HIDDEN_SIZE = 64
LR = 0.001
EPOCHS = 300


class SimpleAnalyser:

    def __init__(self, rnn: SimpleRNN, tokenizer: Tokenizer):
        self.rnn = rnn
        self.tokenizer = tokenizer

    def __call__(self, text: str) -> float:
        X = build_X(sents=[text], tokenizer=self.tokenizer)
        H_last = self.rnn.forward(X)
        Hy = self.rnn.W_hy(H_last)
        Hy = torch.sigmoid(Hy)
        return Hy.item()


def main():
    # --- 데이터 구축 --- #
    sents = [
        sent
        for sent, _ in DATA
    ]
    labels = [
        label
        for _, label in DATA
    ]
    # print(sents)
    # print(labels)
    #
    # 이제 뭘해야하지?
    # 재용님 - 토큰화. &  정수인코딩/
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(texts=sents)  # 정수 인코딩을 학습.
    # seqs = tokenizer.texts_to_sequences(texts=sents)
    # for seq in seqs:
    #     print(seq)
    # # 다음엔 뭘하지? - 길이를 통일하기 위해 패딩을 진행
    # seqs = pad_sequences(sequences=seqs, padding="post", value=0)
    # for seq in seqs:
    #     print(seq)
    # 패딩까지 했으면.
    # 글자의 크기를 학습한다는 것을 방지하기 위해 - 원핫 or 임베딩.
    # 영학님 - 임베딩.
    # num_embeddings 는 어떤값?
    # 재용님 - 고유한 단어의 개수
    vocab_size = len(tokenizer.word_index.keys())
    # 이건 왜? 왜 어휘의 크기를 하나 더 늘려주어야 하는가?
    vocab_size += 1  # padding token = 0. 이것도 추가.


    # 배치를 만들어보기.
    # 1. 커스텀 Dataset 정의하기
    # 2. Dataloader 만들기.
    X = build_X(sents, tokenizer)
    y = build_y(labels)
    dataset = SimpleDataset(X, y)
    # print(dataset[:3])
    # print(len(dataset))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    rnn = SimpleRNN(vocab_size, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE)

    optimizer = torch.optim.Adam(params=rnn.parameters(), lr=LR)

    # 각 배치를 돌면서, 로스를 계산.
    for epoch in range(EPOCHS):
        losses = list()
        for b_idx, batch in enumerate(loader):
            X, y = batch
            # 재용님 - 모델학습.
            loss = rnn.training_step(X, y)
            # 로스를 계산했다. 그럼 뭘해야될까?
            loss.backward()  # 오차역전파.
            # 그럼 이제 모든 계산 그래프 속의 노드에 해당하는 기울기를 구했다.
            # 지현님 -경사하강법.
            optimizer.step()
            optimizer.zero_grad()  # 기울기가 축적되지 않도록 역전파된 기울기를 리셋.
            losses.append(loss.item())
        avg_loss = (sum(losses) / len(losses))
        print(epoch, "->", avg_loss)

    analyser = SimpleAnalyser(rnn, tokenizer)

    print("--- training data 테스트 ----")
    for sent, label in DATA:
        print(sent, label, analyser(sent))

    print("나는 자연어처리가 좋아", analyser("나는 자연어처리가 좋아"))


if __name__ == '__main__':
    main()
