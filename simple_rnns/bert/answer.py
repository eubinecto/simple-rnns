
from typing import List, Tuple
from transformers import BertTokenizer, BertModel
import torch
from torch.nn import functional as F


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


def load_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def build_X(sents: List[str], tokenizer: BertTokenizer, device: torch.device) -> torch.Tensor:
    encodings = tokenizer(text=sents,
                          add_special_tokens=True,
                          return_tensors='pt',
                          truncation=True,
                          padding=True)
    return torch.stack([
        encodings['input_ids'],
        encodings['token_type_ids'],
        encodings['attention_mask']
    ], dim=1).to(device)


def build_y(labels: List[int], device: torch.device) -> torch.Tensor:
    return torch.FloatTensor(labels).unsqueeze(-1).to(device)


class SentimentClassifier(torch.nn.Module):
    def __init__(self, bert: BertModel):
        super().__init__()
        self.bert = bert
        self.hidden_size = bert.config.hidden_size
        self.W_hy = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, 3, L)
        :return: H_all (N, L, H)
        """
        input_ids = X[:, 0]
        token_type_ids = X[:, 1]
        attention_mask = X[:, 2]
        H_all = self.bert(input_ids, token_type_ids, attention_mask)[0]
        return H_all

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X:
        :return:
        """
        H_all = self.forward(X)
        H_cls = H_all[:, 0]
        y_hat = self.W_hy(H_cls)  # (N, H) * (H, 1) -> (N, 1)
        return torch.sigmoid(y_hat)

    def training_step(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_hat = self.predict(X)
        loss = F.binary_cross_entropy(y_hat, y).sum()
        return loss


class Analyser:
    """
    lstm기반 감성분석기.
    """
    def __init__(self, classifier: SentimentClassifier, tokenizer: BertTokenizer, device: torch.device):
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, text: str) -> float:
        X = build_X(sents=[text], tokenizer=self.tokenizer, device=self.device)
        y_hat = self.classifier.predict(X)
        return y_hat.item()

# --- hyper parameters --- #
EPOCHS = 20
LR = 0.0001


def main():
    device = load_device()
    tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-base')
    bert = BertModel.from_pretrained('beomi/kcbert-base')
    sents = [sent for sent, _ in DATA]
    labels = [label for _, label in DATA]

    # --- have a look at the config --- #
    print(bert.config)

    # --- tokenization --- #
    encoded = tokenizer(text=sents, add_special_tokens=True, padding=True, truncation=True)
    print(encoded['input_ids'])
    print(encoded['token_type_ids'])
    print(encoded['attention_mask'])

    # build the d
    X = build_X(sents, tokenizer, device)
    y = build_y(labels, device)

    classifier = SentimentClassifier(bert)
    classifier.train()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        loss = classifier.training_step(X, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"epoch:{epoch}, loss:{loss.item()} ")

    classifier.eval()
    analyser = Analyser(classifier, tokenizer, device)

    for sent in TESTS:
        print(sent, "->", analyser(sent))


if __name__ == '__main__':
    main()
