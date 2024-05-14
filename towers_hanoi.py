
def hanoi(n, source, target, auxiliary):
    if n == 1:
        print('Move disk 1 from source', source, 'to target', target)
        return
    hanoi(n-1, source, auxiliary, target)
    print('Move disk', n, 'from source', source, 'to target', target)
    hanoi(n-1, auxiliary, target, source)
n = 10
hanoi(n, 'A', 'C', 'B')


