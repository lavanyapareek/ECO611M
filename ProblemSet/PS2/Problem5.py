'''
Write a code to write a dictionary into a .txt file. For example, if the
dictionary is
{“Name”:[“abc”,“def”,“ghi”],“Mark 1”:[7,8,9],“Mark 2”:[6,3,8]},
the code should print the following in the text file:
Name=[abc,def,ghi]
Mark 1=[7,8,9]
Mark 2=[6,3,8]
(Hint: The command df.columns gives the name of the columns in the
dataframe df.)
'''
import pandas as pd

dict = {
    "Name": ["abc","def","ghi"],
    "Mark 1":[7,8,9],
    "Mark 2":[6,3,8]
}
df = pd.DataFrame(dict)
with open("txt.txt", 'w') as file:
    for column in df.columns:
        file.write(f"{column} = {list(df[column])}\n")
