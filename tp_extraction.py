# target price report

#https://www.sl886.com/stock/list?list=ratingnews&code=00700

import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import os
import sys

#-----------------------------------------------------------------------------------

save_dir = '/Users/davidlao/Stock_Project'
# 0700.HK, 0823.HK
if len(sys.argv) > 1:
    ticker = sys.argv[1]
else:
    ticker = "0700.HK"


defined_col = ['Date', 'Org', 'Suggest', 'Target_Price', 'Movement', 'Potential']
#-----------------------------------------------------------------------------------

ticker_text = "0" + ticker[:-3]

if not os.path.exists(save_dir + '/' + ticker):
    os.mkdir(save_dir + '/' + ticker)
    print('Created stock folder')

def construct_table(target_URL):
    
    for page_no in range(1, 10):
        
        URL = target_URL + '&page=' + str(page_no)
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, 'html.parser')
        tb = soup.find('table', class_ = "table table-striped table-bordered")
          
        col_list = []
        # for header, last element useless = source
        for i in tb.find_all('th')[:-1]:
            col_list.append(i.get_text())
            
        output_table = pd.DataFrame(columns = col_list)
        if page_no == 1:
            final_output_table = output_table.copy()
        
        for i in tb.find_all('tr')[1:]:
            data_list = []
            for j in i.find_all('td'):
                data_list.append(j.get_text())
            data_list = data_list[:-1]
            output_table.loc[len(output_table)] = data_list
            
        #print(output_table.shape)
        #print(final_output_table.shape)
        
        # check if stop, and combine

        if page_no >= 2 and (output_table.values == final_output_table[-output_table.shape[0]:].values).all():
            print('Stop at page {0}'.format(page_no))
            break
        else:
            final_output_table = final_output_table.append(output_table)
        
    return final_output_table

table = construct_table('https://www.sl886.com/stock/list?list=ratingnews&code=' + ticker_text)

table.columns = defined_col

if os.path.exists(save_dir + '/' + ticker + '/' + 'Target_price' + '.csv'):
    old_table = pd.read_csv(save_dir + '/' + ticker + '/' + 'Target_price' + '.csv')
    table = table[old_table.columns]
    table = table.append(old_table)
    table.drop_duplicates(subset=['Date', 'Org'], keep = 'last', inplace = True)
    
table = table.sort_values('Date', ascending = False)
table.to_csv(save_dir + '/' + ticker + '/' + 'Target_price' + '.csv', index = None, encoding = 'utf-8-sig')

