
from keras_preprocessing.text import Tokenizer


def main():
    sent = "난 너를 좋아해"
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(sent)  # 정수인코딩 학습
    seqs = tokenizer.texts_to_sequences(texts=[sent])
    print(seqs)


if __name__ == '__main__':
    main()