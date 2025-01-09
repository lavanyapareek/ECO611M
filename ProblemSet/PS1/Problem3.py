'''
Question 3 :
Write a code to compute all the saddle points of a given matrix along with
the locations in the matrix. Recall that a saddle point is an element of a
matrix that is simultaneously the maximum of a row and the minimum of
a column, or the minimum of a row and the maximum of a column.
'''
#Brute Force Method:
import numpy as np
#X = [[1,10,4,2],[9,3,8,7],[15,16,17,12]]
X = [[1,1,1],[1,1,1],[1,1,1]]

for i in range(len(X)):
    np.max(X[i])

def saddle(matrix):
        N = len(matrix)
        M = len(matrix[0])
        rowMin = []
        for i in range(N):
            rMin = float('inf')
            for j in range(M):
                rMin = min(rMin, matrix[i][j])
            rowMin.append(rMin)
        colMax = []
        for i in range(M):
            cMax = float('-inf')
            for j in range(N):
                cMax = max(cMax, matrix[j][i])
            colMax.append(cMax)
        saddlePoints = []
        for i in range(N):
            for j in range(M):
                if matrix[i][j] == rowMin[i] and matrix[i][j] == colMax[j]:
                    saddlePoints.append([i,j])
        return saddlePoints
print(saddle(X))
print(saddle(np.transpose(X)))

#Time Complexity : O(NM)
#Space Complexity : O(N + M)

#Greedy Algorithim : 

def greedySaddle(matrix):
    N, M = len(matrix), len(matrix[0])

    r_min_max = float('-inf')
    for i in range(N):
        r_min = min(matrix[i])
        r_min_max = max(r_min, r_min_max)
    
    c_max_min = float('inf')
    for i in range(M):
        c_max = max(matrix[j][i] for j in range(N))
        c_max_min = min(c_max_min, c_max)
    if c_max_min == r_min_max :
        return [c_max_min]
    else:
        return []
print(greedySaddle(X))

#Time Complexity : O(NM)
#Space Complexity : O(1)