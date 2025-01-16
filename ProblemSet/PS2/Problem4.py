'''
Write a code to print the longest common substring in two given strings.
For example, if the strings are “conic section” and “carbonic acid”, your
code should print “onic ” with a space at the end.
'''

s1 = "conic section"
s2 = "carbonic acid"

res = 0
start = -1
end = -1
m = len(s1)
n = len(s2)

for i in range(m):
    for j in range(n):
        curr = 0
        while i + curr < m and j + curr < n and s1[i + curr] == s2[j + curr]:
            curr += 1
            resc = res
            res = max(res, curr)
            if(res > resc):
                start = i
                end = i + curr
print(s1[start:end])


#########DP Solution with memoisation taking less space???? ##########

