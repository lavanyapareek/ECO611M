'''
Write a code to compute nCr using recursion. Do this without computing
the factorial of any number.
'''
'''
nCr = n!/(r!*(n - r)!) = [(n - (r - 1))*(n - (r - 2))*(n - (r - 3))*...*1]/[(r)*(r-1)*...*1]
'''

def choose(n, r):
    if n < r:
        return -1
    if n == r:
        return 1
    if r == 0:
        return 1
    return ((n - (r - 1))/r)*choose(n, r - 1)

print(choose(4, 2))