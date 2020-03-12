def IsMatch(strategy,seqA,seqB):
    new_seqA = []
    bak_seqA = list(seqA)
    for s in strategy:
        if s == 'd':
            bak_seqA = bak_seqA[1:]

        if s == 'l':
            new_seqA = [bak_seqA[0]] + new_seqA
            bak_seqA = bak_seqA[1:]

        if s == 'r':
            new_seqA = new_seqA + [bak_seqA[0]]
            bak_seqA = bak_seqA[1:]

    if isEqual(new_seqA, seqB):
        return True
    else:
        return False

def isEqual(seqA,seqB):

    if len(seqA) != len(seqB):
        return False

    for i in range(0,len(seqA)):
        if seqA[i] != seqB[i]:
            return False

    return True

def ProduceStrategy(select,strategy,seqA,seqB,result):
    if len(strategy) == len(seqA):
        if IsMatch(strategy, seqA, seqB):
            result.append(list(strategy))

        return

    for s in select:
        strategy.append(s)
        ProduceStrategy(select, strategy, seqA, seqB, result)
        strategy.pop()



    return

import sys
if __name__ == '__main__':

    select = ['d','l','r']
    # n = int(sys.stdin.readline().strip())
    n = 1

    for i in range(0, n):
        # line1 = input()
        # line2 = input()
        line1 = '123456'
        line2 = '3245'

        seqA = list(map(int, list(line1)))
        seqB = list(map(int, list(line2)))

        strategy = []
        result = []

        ProduceStrategy(select,strategy,seqA,seqB,result)
        print('{')
        for d in result:
            for j in range(0, len(d)):
                if j != len(d) - 1:
                    print(d[j], end=' ')
                else:
                    print(d[j], end='\n')

        print('}')