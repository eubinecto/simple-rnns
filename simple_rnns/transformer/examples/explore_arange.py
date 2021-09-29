import torch


def main():
    # X = torch.randint(high=29, size=(10, 30))  # (N, L)
    # pos_embeddings = torch.nn.Embedding(num_embeddings=X.shape[1], embedding_dim=512)
    # range()
    length = 10
    print(torch.arange(10))


if __name__ == '__main__':
    main()