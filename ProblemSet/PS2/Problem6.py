'''
Write a code to sort the data in a dataframe based on a given column,
without using any in-built command. In the above example, if you sort
the dictionary based on “Mark 2”, then your dataframe becomes
{“Name”:[“def”,“abc”,“ghi”],“Mark 1”:[8,7,9],“Mark 2”:[3,6,8]}.
'''
import pandas as pd

dict = {
    "Name": ["abc","def","ghi"],
    "Mark 1":[7,8,9],
    "Mark 2":[6,3,8]
}
df = pd.DataFrame(dict)

def sort(column, df):
    column_values = df[column].tolist()
    indices = list(range(len(column_values)))
    
    #Michale Buble sort
    for i in range(len(column_values)):
        for j in range(0, len(column_values) - i - 1):
            if column_values[indices[j]] > column_values[indices[j + 1]]:
                indices[j], indices[j + 1] = indices[j + 1], indices[j]
    
    sorted_data = {
        key: [df[key][i] for i in indices] for key in df.columns
    }
    return pd.DataFrame(sorted_data)
df = sort("Mark 2", df)
print(df)