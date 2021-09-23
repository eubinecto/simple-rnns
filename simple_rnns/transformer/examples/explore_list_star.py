

def add(a, b):
    return a + b



def add_too_much(a, b, c, d, e, f, g, h, i, j):
    return a + b + c + d + e + f + g + h + i + j

def main():
    print(add(1, 2))
    # print(add([1, 2]))  # 오류
    print(add(*[1, 2]))
    print(add(**{'a': 1, 'b': 2}))
    # print(add(a=1, b=/))
    params = list(range(10))
    print(add_too_much(*params))


if __name__ == '__main__':
    main()