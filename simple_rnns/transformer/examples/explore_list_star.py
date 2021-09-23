

def add(a, b):
    return a + b


def main():
    print(add(1, 2))
    print(add(*[1, 2]))

if __name__ == '__main__':
    main()