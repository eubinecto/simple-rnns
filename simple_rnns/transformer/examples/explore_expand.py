import torch


def main():
    X = torch.randint(high=29, size=(10, 30))  # (N, L)
    X_pos = torch.arange(X.shape[1]).expand(X.shape[0],  X.shape[1])  # (N, L) -> (N, L)
    pos_embeddings = torch.nn.Embedding(num_embeddings=X.shape[1], embedding_dim=512)
    X_pos = pos_embeddings(X_pos)
    print(X_pos.shape)
    print(X_pos)


if __name__ == '__main__':
    main()
