from konlpy.tag import Hannanum
# zulu - macOS/arm64/version15 설치 이후.
# 해당 path를 지정.
JVM_PATH = '/Library/Java/JavaVirtualMachines/zulu-15.jdk/Contents/Home/bin/java'


def main():
    okt = Hannanum(jvmpath=JVM_PATH)
    tokens = okt.morphs("나는 너를 좋아해")
    print(tokens)
    tokens = okt.morphs("나는 너를 좋아하나")
    print(tokens)
    tokens = okt.morphs("나는 너를 좋아할수도")
    print(tokens)

if __name__ == '__main__':
    main()
