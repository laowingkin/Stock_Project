#stocks = ['HSI']
#sn = StockNews(stocks, wt_key='MY_WORLD_TRADING_DATA_KEY')
#sn = StockNews(stocks = ['^HSI'], wt_key='9QewcQsCTMRyhtqJXKck2vCMwLhsmnOmD9rK5OVfEvHKUWUdHL1QrzZqtNPE')
#news = sn.summarize()
#news

#https://newsapi.org
#0cbaf8804b05479bab8e7c0e88cbb5f1

from GoogleNews import GoogleNews
import pandas as pd
import numpy as np
import re
from datetime import datetime
from googletrans import Translator
from googleapiclient.discovery import build
import os
import sys
#AIzaSyBBc_9Bu5W9tAI67x99XhzDL74ndacU3VA

# HK VPN for dates!
# parameters
# -------------------------------------------------------------------------------------------
save_dir = '/Users/davidlao/Stock_Project'
# 0700.HK, 0823.HK
#ticker = "0700.HK"
if len(sys.argv) > 1:
    ticker = sys.argv[1]
else:
    ticker = "0700.HK"
    
if ticker == "0700.HK":
    other_desc = ["tencent", ".hsi"]
elif ticker == "0823.HK":
    other_desc = ["link reit", ".hsi"]
elif ticker == "0066.HK":
    other_desc = ["mtr", ".hsi"]
else:
    other_desc = []
    
Page_start = 1
Page_until = 5
tolerate_zero = 3
supp_daily_limit_reach = False
# -------------------------------------------------------------------------------------------

# for daily run: fewer pages
key_word = [ticker] + other_desc

# for historical
#key_word = [key_word[2]]

if not os.path.exists(save_dir + '/' + ticker):
    os.mkdir(save_dir + '/' + ticker)
    print('Created stock folder')

text = pd.DataFrame(columns = ['title', 'date'])

for k in key_word:
    print('--- for {0} ---'.format(k))
    tolerate_cnt = 0
    for i in range(Page_start, Page_until):
        googlenews = GoogleNews('en', 'd')    # to avoid limit?
        googlenews.search(k)    # will perform getpage(1)
        googlenews.getpage(i)
        temp_size = text.shape[0]
        for j in googlenews.result():
            temp = pd.DataFrame([[j['title'], j['date']]], columns = ['title', 'date'])
            text = text.append(temp)
            text.drop_duplicates(subset=['title'], keep = 'last', inplace = True)
        print('# added articles on P.{0} :{1}'.format(i, text.shape[0] - temp_size))
        if text.shape[0] - temp_size == 0:
            tolerate_cnt+=1
        if tolerate_cnt >= tolerate_zero:
            print('Break due to consistently no new articles returned')
            break
              
    print('# batch articles after {0}: {1}'.format(k, text.shape[0]))


# convert to english date
re_year = re.compile(r'([0-9]{4})年.*?')
re_month = re.compile(r'年([0-9]{1,2})月.*?')
re_day = re.compile(r'月([0-9]{1,2})日')
text['Year'] = text['date'].apply(lambda x: re_year.findall(x)[0] if len(re_year.findall(x)) > 0 else 0)
text['Month'] = text['date'].apply(lambda x: re_month.findall(x)[0] if len(re_month.findall(x)) > 0 else 0)
text['Day'] = text['date'].apply(lambda x: re_day.findall(x)[0] if len(re_day.findall(x)) > 0 else 0)

def combine_date(x):
    if x['Year'] == 0:
        return datetime.now().date()
    else:
        return datetime(int(x['Year']), int(x['Month']), int(x['Day']))
    
text['Eng_date'] = text.apply(lambda x: combine_date(x), axis = 1)
text = text.sort_values('Eng_date', ascending = False)

# for placeholder
text['supp'] = ''
text['title_EN'] = ''
text['supp_EN'] = ''

# for ordering
text = text[['date', 'Year', 'Month', 'Day', 'Eng_date', 'title', 'title_EN', 'supp', 'supp_EN']]

# load historical
if  os.path.exists(save_dir + '/' + ticker + '/' + 'Stock_news' + '.csv'):
    old_text = pd.read_csv(save_dir + '/' + ticker + '/' + 'Stock_news' + '.csv')
    text = text[old_text.columns]
    text = text.append(old_text)
    text.drop_duplicates(subset=['title'], keep = 'last', inplace = True)

text['Eng_date'] = pd.to_datetime(text['Eng_date'])
text = text.sort_values('Eng_date', ascending = False)
text.to_csv(save_dir + '/' + ticker + '/' + 'Stock_news' + '.csv', index = None, encoding = 'utf-8-sig')
print('# total articles in master: {0}'.format(text.shape[0]))


# supplement from google custom search
def custom_search_supp(file_path, daily_limit = 500):
    file_text = pd.read_csv(file_path)
    cnt = 0
    for i in range(file_text.shape[0]):
        if cnt >= daily_limit:
            return file_text
        if (file_text['supp'].iloc[i] == '') or (pd.isnull(file_text['supp'].iloc[i])):
            service = build("customsearch", 'v1', developerKey='AIzaSyBBc_9Bu5W9tAI67x99XhzDL74ndacU3VA')
            try:
                file_text['supp'].iloc[i] = service.cse().list(q=file_text['title'].iloc[i], 
                         cx='013639151591574449208:a7keah7pj48').execute()['items'][0]['snippet']
            except:
                file_text['supp'].iloc[i] = file_text['title'].iloc[i]
            cnt += 1
    print('Supplemented {0} records'.format(cnt))
    return file_text

if supp_daily_limit_reach == False:
    supp_text = custom_search_supp(save_dir + '/' + ticker + '/' + 'Stock_news' + '.csv')
    supp_text.to_csv(save_dir + '/' + ticker + '/' + 'Stock_news' + '.csv', index = None, encoding = 'utf-8-sig')

# translation: unlimited
def cn_tran_en(file_path):
    file_text = pd.read_csv(file_path)
    cnt = 0
    for i in range(file_text.shape[0]):
        # only translate if empty
        if (file_text['title_EN'].iloc[i] == '') or (pd.isnull(file_text['title_EN'].iloc[i])):
            translator = Translator()
            file_text['title_EN'].iloc[i] = translator.translate(file_text['title'].iloc[i]).text
            cnt += 1
        # only translate if empty EN and has supp
        if (file_text['supp_EN'].iloc[i] == '') or (pd.isnull(file_text['supp_EN'].iloc[i])):
            if (not file_text['supp'].iloc[i] == '') and (not pd.isnull(file_text['supp'].iloc[i])):
                translator = Translator()
                file_text['supp_EN'].iloc[i] = translator.translate(file_text['supp'].iloc[i]).text
                cnt += 1
    print('Translated {0} records'.format(cnt))
    return file_text

trans_text = cn_tran_en(save_dir + '/' + ticker + '/' + 'Stock_news' + '.csv')
trans_text.to_csv(save_dir + '/' + ticker + '/' + 'Stock_news' + '.csv', index = None, encoding = 'utf-8-sig')

# for backup
if not os.path.exists(save_dir + '/' + ticker + '/backup'):
    os.mkdir(save_dir + '/' + ticker + '/backup')
trans_text.to_csv(save_dir + '/' + ticker + '/backup/' + 'Stock_news_' + str(datetime.now().date()) + '.csv', index = None, encoding = 'utf-8-sig')


