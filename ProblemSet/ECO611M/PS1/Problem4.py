'''
Write a code to compute the product of two polynomials a0 + a1x +
a2x2 +· · · + anxn and b0 + b1x + b2x2 +· · · + bmxm. In other words, given
two vectors [a0, a1, . . . , an] and [b0, b1, . . . , bm], your code should print the
product vector of length (n + m + 1).

(a0 + a1x)*(b0 + b1x) = a0b0 + (a0b1 + b0a1)x + a1b1x^2
'''
import numpy as np

A = np.array([1,2,3,4,5,6,7,8,9])
B = np.array([3,4,2,3,4,5,6,7,8,9])
#3 + 10x +  8x^2


res = [0]*(len(A) + len(B) - 1)

for i in range(len(A)):
    for j in range(len(B)):
        res[i + j] += A[i]*B[j]

result = ''
for i in range(len(res)):
    result = result + str(res[i]) + 'x' + str(i) + ' ' + '+' + ' '
print(result)