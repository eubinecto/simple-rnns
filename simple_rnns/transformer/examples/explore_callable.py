from typing import List, Callable


def process(l: List[int], func: Callable) -> List[int]:
    return [
        func(e)
        for e in l
    ]


def main():
    my_list = [1, 2 , 3]
    print(process(my_list, lambda x: x**2))  # 타입힌팅을 함수로 할 수도 있따.
    print(process(my_list, "유빈"))  #


if __name__ == '__main__':
    main()