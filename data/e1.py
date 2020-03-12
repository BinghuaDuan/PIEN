def SortedByLevel(array,N):
    bound = Partition(array)
    high = array[0:bound]
    low = array[bound:]

    high.sort(reverse=True)
    low.sort(reverse=True)

    result = high + low

    return result[0:N]

def Partition(array):
    i = -1
    for j in range(0, len(array)):
        if (array[j] & 1) == 0:
            array[i+1], array[j] = array[j], array[i+1]
            i += 1
    return i + 1

import sys
if __name__ == "__main__":

    line = sys.stdin.readline().strip()
    [array, N] = line.split(";")
    array = list(map(int, array.split(",")))
    N = int(N)
    ans = SortedByLevel(array, N)
    for i in range(0, len(ans) - 1):
        print(ans[i], end=',')
    print(ans[len(ans) - 1])
