'''
For a given string, write a code to print the number of lower case alpha-
bets (“a-z”), upper case alphabets (“A-Z”), numbers (“0-9”), and other
characters (space bar, symbols, etc.).
'''

string = "bbbb bfdjghdfbj djfhgbdhlsfgiu4397r 3qwjbn frd32r809u2 0129389q33q9q3094u ksjbdv cxm'pAWOieqwldf.sf  \t \n KJDFHG7EWR E89UQ3E OIWERU -QURFSDXMNC WEPOR -Q20EIQ39 EWP9UR "

res = {}
for char in string:
    if char in res:
        res[char] += 1
    else:
        res[char] = 1
print(res)