'''
Question 2 : Write a code to compute the greatest common 
factor and the least common multiple of two numbers.
'''
x = 2**10 * 3**5 * 5**25
y = 2**8 * 3**6 * 7**4

##------------------------------GCD------------------------------##

##Divisiors method
'''
facx = []
facy = []

for i in range(1, x+1):
    if x % i == 0:
        facx.append(i)

for i in range(1, y+1):
    if y % i == 0:
        facy.append(i)


print(f"Factors of {x}: {facx}")
print(f"Factors of {y}: {facy}")

idxx = len(facx) - 1
idxy = len(facy) - 1

while idxx >= 0 and idxy >= 0:
    if facx[idxx] == facy[idxy]:
        print(f"Common factor: {facx[idxx]}")
        break
    elif facx[idxx] > facy[idxy]:
        idxx -= 1
    else:
        idxy -= 1
'''
#Time complexity : O(x + y)
#Space complexity : O(log(x) + log(y))

##---------------------------------------------------------------##

#Prime factor Method :
'''
primeX = {}
for i in range(2, int(x**0.5) + 1):
    while x%i == 0:
        if i not in primeX:
            primeX[i] = 0
        primeX[i] += 1
        x = x/i
if x > 1:
    primeX[x] = 1

primeY = {}
for i in range(2, int(y**0.5) + 1):
    while y%i == 0:
        if i not in primeY:
            primeY[i] = 0
        primeY[i] += 1
        y = y/i
if y > 1:
    primeY[y] = 1
print(primeX)
print(primeY)

gcd = 1
for prime in primeX :
    if prime in primeY:
        min_exp = min(primeX[prime], primeY[prime])
        gcd *= prime**min_exp
print(gcd)
'''
#Time Complexity : O(sqrt(x)log(x) + sqrt(y)log(y))
#Space Complexity : O(log(x) + log(y))

##---------------------------------------------------------------##

#Euclid's Algorithim : 
'''
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
print(gcd(y,x))
'''
#Time complexity : O(log(min(a,b)))
#Space complexity : O(1)

##---------------------------------------------------------------##


#Binary GCD(Stien's Algorithm):
'''
def gcd(a,b):
    if a == b:
        return a
    if a == 0:
        return b
    if b == 0:
        return a
    
    if (a & 1) == 0 and (b & 1) == 0:
        return gcd(a >> 1, b >> 1) << 1
    elif a & 1 == 0:
        return gcd(a >> 1, b)
    elif b & 1 == 0:
        return gcd(a, b >> 1)
    elif a > b :
        return gcd((a - b) >> 1, b)
    else:
        return gcd(a, (b - a) >> 1)
print(gcd(x,y))
'''
#Time Complexity : O(log(min(x, y)))
#Space Complexity : O(log(min(x, y))) (Recursive tree getting stored in memory)

##---------------------------------------------------------------##

##------------------------------LCM------------------------------##
'''
Method 1:
LCM = x*y/GCD
Calculate GCD any way.
'''    

##---------------------------------------------------------------##

#Prime Factor Method :


