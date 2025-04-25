'''
Write a code to find if two strings are permutations of each other. For
example, “silent” and “listen” are permutations; “you are” and “are you”
are permutations; but “loot” and “lot” are not permutations.
'''
from collections import Counter
string1 = "wubbalubbadubdub"
string2 = "dubdubwubbalubba"

def are_anagrams_bitwise(string1, string2):
    if len(string1) != len(string2):
        return False

    xor_result = 0
    for char in string1 + string2:
        xor_result ^= ord(char)

    return xor_result == 0
print(are_anagrams_bitwise(string1, string2))
###########################
if Counter(string1) == Counter(string2):
    print(True)
else :
    print(False)

###########################
string1 = sorted(string1)
string2 = sorted(string2)

if string1 == string2 :
    print(True)
else:
    print(False)

