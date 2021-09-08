
import torch


def main():
    # (20, 30) 의 사이즈를 가지는 가중치 (더하기 편향) 행렬
    W = torch.nn.Linear(in_features=20, out_features=30)
    # (5, 20) 사이즈를 가지는 X
    X = torch.randn(size=(5, 20))
    # 가중치를 행렬곱 해준 것.
    result = W(X)  # (5, 20) * (20, 30)  -> (5, 30)
    print(result.shape)



if __name__ == '__main__':
    main()