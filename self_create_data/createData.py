"""Teaching ML: Students
Here we simply create the data and export to the excel
zero: product id
first: sales on jan
second: sales on Feb
third: sales on march
fourth: demand rate: very_high, high, medium,low,very_low
fifth: highest_seller: Company name from A to D
target: to_prodict: sales on April
"""

import sys
sys.path.append('C:\\User\\pande\\Documents\\Python_packages')
import numpy as np
import pandas as pd
import random

# create the empty dataframe with 7 column 1000 data
# data generator
def rondom_generator(n, n1, n2):
    ran = range(n1, n2)
    count = random.sample(ran, 400)
    elem = np.random.choice(count, n)
    return elem
# values list
seller_list =['A', 'M', 'L', 'D']
demand = ['veryHigh', 'high','medium','low','veryLow']


data = {'productId':rondom_generator(n=2000, n1=10000, n2=999999),'sales1':rondom_generator(n=2000, n1=20, n2=489),
        'sales2': rondom_generator(n=2000, n1=10, n2=530), 'sales3': rondom_generator(n=2000, n1=100, n2=649),
        'sales4': rondom_generator(n=2000, n1=100, n2=860), 'deman_level': random.choices(demand, k=2000),
        'best_seller':random.choices(seller_list, k=2000), 'target': rondom_generator(n=2000, n1=50, n2=999) }
df = pd.DataFrame(data)
print(df.head())
# exporting to excel:
path_data =r'C:\Users\pande\PycharmProjects\LinearREgression_OwnData\sales_data.xlsx'
df.to_excel(path_data, index=False, header=True)




