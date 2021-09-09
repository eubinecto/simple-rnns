import torch


def main():
    my_list = [1, 2, 3]
    a = torch.Tensor(my_list)
    b = torch.Tensor(my_list)
    # 저렇게 만든 텐서를 가지고서 연산을 하면,
    # 그러면 텐서 객체 안에 어떤 연산을 거쳤는지가 저장. - 그래서 텐서 객체를 사용한다.
    c = a + b
    print(c)


if __name__ == '__main__':
    main()