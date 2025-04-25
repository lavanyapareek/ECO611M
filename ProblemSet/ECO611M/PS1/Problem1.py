'''
Write a code to obtain a positive integer as the input, and spell the number
digit-by-digit. For example, if the input is 4503, your code should print
“four five zero three” (in the correct order).
'''
x = input("Enter a number :")
def digitToWord(c):
    match c:
        case '0':
            return 'Zero'
        case '1':
            return 'One' 
        case '2':
            return 'Two'
        case '3':
            return 'Three'
        case '4':
            return 'Four'
        case '5':
            return 'Five'
        case '6':
            return 'Six'
        case '7':
            return 'Seven'
        case '8':
            return 'Eight'
        case '9':
            return 'Nine'
        
res = ''
for i in range(len(x)):
    res += ' ' + digitToWord(x[i])
print(res)