'''
Write a code to print the number of characters in each word of a given
string.
'''

string = "I am a stupid boy who just lost his wallet and all its contents including Rs.3000, Government IDs and Bank Cards."

strlist = string.split(' ')

res = {}

for word in strlist:
    res[word] = len(word)
print(res)