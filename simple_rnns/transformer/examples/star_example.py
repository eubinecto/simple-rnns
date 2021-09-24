

def add(a, b):
    return a + b


def main():
    print(add(1, 2))
    args = list(range(2))
    print(args)
    print(add(*args))


if __name__ == '__main__':
    main()