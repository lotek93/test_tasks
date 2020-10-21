# since using of modules is not allowed in this task we cannot use reduce() in python3
# from functools import reduce

def multiplicate(A):
    B = []
    for i in range(len(A)):
        a = A.copy()
        del a[i]
#        b = reduce(lambda x, y: x * y, a, 1)
        b = 1
        for j in range(len(a)):
            b *= a[j]
        B.append(b)
    return B


A = [1, 2, 3, 4]
assert multiplicate(A) == [24, 12, 8, 6]
