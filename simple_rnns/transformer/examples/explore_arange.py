import torch

def main():
    X = torch.randint(high=29, size=(10, 30))
    pos_embeddings = torch.nn.Embedding(num_embeddings=X.shape[1], embedding_dim=512)
    X_pos = torch.arange(start=0, end=X.shape[1] - 1).repeat(X.shape[0])
    print(X_pos.shape)


if __name__ == '__main__':
    main()