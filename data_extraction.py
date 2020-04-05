import yfinance as yf
from stocknews import StockNews
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, Conv1D, GlobalMaxPooling1D, TimeDistributed, RepeatVector, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.layers import LeakyReLU
from sklearn.preprocessing import normalize
import keras.backend as K
import keras.objectives
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') 
import os
import time
from datetime import datetime
from datetime import timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from numpy.random import seed
seed(226)
import tensorflow as tf    
tf.random.set_seed(226)
import sys
import holidays

# Full column list
#['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits',
#       'SMA_10', 'Close_below_SMA_10', 'SMA_20', 'Close_below_SMA_20',
#       'SMA_50', 'Close_below_SMA_50', 'SMA_100', 'Close_below_SMA_100',
#       'SMA_150', 'Close_below_SMA_150', '^HSI_Close', 'title_sc', 'supp_sc',
#       'Target_Price', 'BB_Price', 'BB_Count', 'Date', 'Short_Ratio',
#       'Short_Count', 'Short_Amount', 'Target_Price_Uplift', 'BB_Value',
#       'Close_over_BB', 'BB_Conf']

# TBD add features
# day of week                                             --- ticked
# month start / end                                       --- ticked
# holidays                                                --- ticked
# annual report
# % before hitting certain price point, e.g. $310     
# % between max - min in last x days                      --- ticked
# other sentiment methods
# more complete supp text
# today high < yesterday low / today low > yesterday high ---ticked

# parameters
# -------------------------------------------------------------------------------------------
if len(sys.argv) >= 2:
    purpose = sys.argv[2]
else:
    purpose = "train"

#purpose = 'Train' / 'Predict'
save_dir = '/Users/davidlao/Stock_Project' 
model_remark = 'self stock price LSTM - 20N'      # only needed when 1st training
# 0700.HK, 0823.HK

if len(sys.argv) > 1:
    ticker = sys.argv[1]
else:
    ticker = "0700.HK"
    
supp_ticker = "^HSI"
input_period = 20
input_col_list = ['High', 'Volume',
                  'Close_above_SMA_100',
                  supp_ticker + '_Close',
                  'Is_Weekday_2', 'Month_end', 'Next_trade_day', 'PHigh_20',
                  'Very_Bad_Ind', 'Very_Good_Ind']
exclude_pct_list = ['Closed_above', 'Overall_sc', 'Target_Price_Uplift', 
                    'Close_over_BB', 'BB_Conf', 'Short_Ratio', 'Short_Count', 'Short_Amount',
                    'Is_Weekday_2', 'Month_end', 'Next_trade_day', 'PHigh_20',
                    'Very_Bad_Ind', 'Very_Good_Ind']
news_mode = False
output_period_max = 1
output_period_min = 1
val_size = 0.3
test_size = 80
clear_kpi = False

if purpose == 'train':
    dateTimeObj = time.strftime("%Y%m%d-%H%M%S")
    test_live = datetime.now().date() + timedelta(days=1)
else:
    if ticker == "0700.HK":
        dateTimeObj = '20200222-095409' 
    elif ticker == "0823.HK":
        dateTimeObj = '20200207-181629'   # for prediction   
    test_live = datetime(int(dateTimeObj[:4]), int(dateTimeObj[4:6]), int(dateTimeObj[6:8]))    # for live, should be same day as above dateTimeObj

input_col_list = input_col_list + ['Target_Price_Uplift'] + ['Close_over_BB'] + ['BB_Conf'] + ['Short_Ratio'] + ['Short_Count']

if news_mode == True:
    input_col_list = input_col_list + ['Overall_sc'] 
    
if ticker == "0700.HK":
    starting_period = datetime(2017, 1, 1)      # for news...
elif ticker == "0823.HK":
    starting_period = datetime(2016, 1, 1)
else:
    starting_period = datetime(2000, 1, 1)
# -------------------------------------------------------------------------------------------

print("Mode: {0}".format(purpose))

if not os.path.exists(save_dir + '/' + ticker):
    os.mkdir(save_dir + '/' + ticker)
    print('Created stock folder')

stock = yf.Ticker(ticker)

df = stock.history(period="max")



###### feature engineering
# for SMA
for i in [10, 20, 50, 100, 150, 200]:
    df['SMA_' + str(i)] = df['Close'].rolling(window=i).mean()
    #df['SMA_' + str(i)].fillna(method = 'bfill', inplace = True)
    #df['Close_below_SMA_' + str(i)] = np.where(df['Close'] < df['SMA_' + str(i)], 1, 0)
    df['Close_above_SMA_' + str(i)] = df['Close'] / df['SMA_' + str(i)]
    df['Close_above_SMA_' + str(i)].fillna(1, inplace = True)

# for position percentage over past x days
for i in [10, 20, 50, 100, 150, 200]:
    for feat in ['Close', 'High', 'Low']:
        df['P' + feat + '_' + str(i)] = (df[feat] - df[feat].rolling(window=i).min()) / (df[feat].rolling(window=i).max() - df[feat].rolling(window=i).min())
        df['P' + feat + '_' + str(i)].fillna(0.5, inplace = True)


# for supp
df_supp = pd.DataFrame(yf.Ticker(supp_ticker).history(period="max")["Close"])
df_supp.rename(columns = {'Close': supp_ticker + '_Close'}, inplace = True)


###### Getting News
df_news = pd.read_csv(save_dir + '/' + ticker + '/' + 'Stock_news' + '.csv')
df_news_extract = df_news.sort_values('Eng_date', ascending = True).groupby(['Eng_date'])['title_EN'].apply(lambda x: ';'.join(x)).reset_index()
df_supp_extract = df_news.sort_values('Eng_date', ascending = True).groupby(['Eng_date'])['supp_EN'].apply(lambda x: ';'.join(x)).reset_index()

# translate to sentiment score
sia = SentimentIntensityAnalyzer()
df_news_extract['title_sc'] = df_news_extract['title_EN'].apply(lambda x: sia.polarity_scores(x)['compound'])
df_supp_extract['supp_sc'] = df_supp_extract['supp_EN'].apply(lambda x: sia.polarity_scores(x)['compound'])
df_news_extract.set_index('Eng_date', inplace = True)
df_supp_extract.set_index('Eng_date', inplace = True)

###### Getting target price from big org
df_tp = pd.read_csv(save_dir + '/' + ticker + '/' + 'Target_price' + '.csv')
df_tp = df_tp[df_tp.Target_Price.apply(lambda x: x != '-')]
df_tp['Target_Price'] = df_tp['Target_Price'].astype(float)
df_tp_extract = df_tp.sort_values('Date', ascending = True).groupby(['Date']).agg({'Target_Price': 'mean'})

###### Getting buy back information
df_bb = pd.read_csv(save_dir + '/' + ticker + '/' + 'Buyback' + '.csv')
df_bb['BB_Price'] = df_bb['BB_Price'].apply(lambda x: 0 if x == '-' else x)
df_bb['BB_Price'] = df_bb['BB_Price'].astype(float)
df_bb['BB_Count'] = df_bb['BB_Count'].apply(lambda x: locale.atoi(x))
df_bb_extract = df_bb.sort_values('Date', ascending = True).groupby(['Date']).agg({'BB_Price': 'mean', 'BB_Count': 'sum'})

###### Getting short information
def p2f(x):
    return float(x.strip('%'))/100
df_sh = pd.read_csv(save_dir + '/' + ticker + '/' + 'Short' + '.csv')
df_sh['Short_Ratio'] = df_sh['Short_Ratio'].apply(p2f)
df_sh['Short_Count'] = df_sh['Short_Count'].apply(lambda x: locale.atoi(x))
df_sh['Short_Amount'] = df_sh['Short_Amount'].apply(lambda x: locale.atoi(x))
df_sh.set_index('Date', inplace = True)

# combine together
df_combined = pd.merge(df, df_supp, how = 'left', left_index = True, right_index = True)
df_combined = pd.merge(df_combined, df_news_extract['title_sc'], how = 'left', left_index = True, right_index = True)
df_combined = pd.merge(df_combined, df_supp_extract['supp_sc'], how = 'left', left_index = True, right_index = True)
df_combined = pd.merge(df_combined, df_tp_extract['Target_Price'], how = 'left', left_index = True, right_index = True)
df_combined = pd.merge(df_combined, df_bb_extract, how = 'left', left_index = True, right_index = True)
df_combined = pd.merge(df_combined, df_sh[['Short_Ratio', 'Short_Count', 'Short_Amount']], how = 'left', left_index = True, right_index = True)

def ffill_decay(x, factor):
    return x.groupby(x.notnull().cumsum()).apply(lambda y: y.ffill() / factor ** np.arange(len(y)))

# further engineering
df_combined['title_sc'].fillna(method = 'ffill', limit = 1, inplace = True)
df_combined['supp_sc'].fillna(method = 'ffill', limit = 1, inplace = True)

df_combined['Target_Price'].fillna(method = 'ffill', limit = 60, inplace = True)
df_combined['Target_Price_Uplift'] = df_combined['Target_Price'] / df_combined['Close'] - 1
#df_combined['Target_Price_Uplift'].fillna(method = 'ffill', limit = 20, inplace = True)
df_combined['Target_Price_Uplift'].fillna(0, inplace = True)

df_combined['Overall_sc'] = (df_combined['title_sc'] + df_combined['supp_sc']) / 2

df_combined['BB_Price'] = np.where(df_combined['BB_Price'] > 0, 
                                   df_combined['BB_Price'], 
                                   np.where(df_combined['BB_Count'] > 0, df_combined['Close'], 
                                            df_combined['BB_Price']))
df_combined['Avg_BB_Price'] = (df_combined['BB_Price'] * df_combined['BB_Count']).cumsum() / df_combined['BB_Count'].cumsum()
df_combined['Avg_BB_Price'] = df_combined['Avg_BB_Price'].fillna(method = 'ffill')
df_combined['Close_over_Avg_BB'] = df_combined['Close'] / df_combined['Avg_BB_Price'] - 1
df_combined['Close_over_Avg_BB'].fillna(0, inplace = True)
df_combined['BB_Value'] = df_combined['BB_Price'].fillna(method = 'ffill')
df_combined['Close_over_BB'] = df_combined['Close'] / df_combined['BB_Value'] - 1
df_combined['Close_over_BB'].fillna(0, inplace = True)

#df_combined['BB_Conf'] = df_combined['BB_Count'].fillna(0)
df_combined['BB_Conf'] = ffill_decay(df_combined['BB_Count'], 1.3)
df_combined['BB_Conf'].fillna(0, inplace = True)


# further feature engineering
for i in range(0, 5):   # 0 = Monday
    df_combined['Is_Weekday_' + str(i)] = np.where(df_combined.index.dayofweek == i, 1, 0)
    
df_combined['Month_end'] = np.where(df_combined.index.day > 27, 1, 0)

def count_next_trade_day(x):
    current_date = x
    hk_holidays = holidays.HongKong()
    cnt = 0
    current_date = current_date + timedelta(days=1)
    while True:       
        if (current_date.weekday() < 5) and (not current_date in hk_holidays):
            return cnt
        else:
            cnt += 1
            current_date = current_date + timedelta(days=1)

df_combined = df_combined.assign(Next_trade_day = pd.Series(df_combined.reset_index()['Date'].apply(lambda x: count_next_trade_day(x))).values)

# compare high & low today vs ytd
df_combined['Very_Bad_Ind'] = 0
df_combined['Very_Good_Ind'] = 0
for i in range(1, df_combined.shape[0]):
    if df_combined['High'].iloc[i] < df_combined['Close'].iloc[i-1]:
        df_combined['Very_Bad_Ind'].iloc[i] = 1
    elif df_combined['Low'].iloc[i] > df_combined['Close'].iloc[i-1]:
        df_combined['Very_Good_Ind'].iloc[i] = 1




# for all normalization columns
for i in ['BB_Conf', 'Short_Count', 'Short_Amount', 'Next_trade_day']:
    df_combined[i] = (df_combined[i]-df_combined[i].min())/(df_combined[i].max()-df_combined[i].min())


# start from certain data mature
df_combined = df_combined[df_combined.index >= starting_period]

# check corr
corr = df_combined[input_col_list].corr()

# for both train & predict
# penalize wrong pred > actual more
def customLoss(true,pred):
    diff = pred - true
    
    greater = K.greater(diff,0)
    greater = K.cast(greater, K.floatx()) #0 for lower, 1 for greater
    #greater = greater + 1                 #1 for lower, 2 for greater
    greater = K.switch(greater > 0, lambda: greater + 2, lambda: greater + 1)
    
    # MSE
    return K.mean(greater*K.square(diff))



def data_transformer(stock, data, input_col_list, input_period, output_period_max, output_period_min, for_train = True):
    print('Stock: {0}'.format(stock))
    print('Data period studied: {0} to {1}'.format(data.index[0].date(), data.index[-1].date()))
    
    # convert to percentage change
    new_data = data.copy()
    #for i in new_data.columns:
    #    if ('Closed_below' not in i) and ('title_sc' not in i): 
    #        new_data[i] = new_data[i].pct_change(periods = 1)
    
    pct_col = [col for col in new_data if not col.startswith(tuple(exclude_pct_list))]
    new_data[pct_col] = new_data[pct_col].pct_change(periods = 1)
    new_data.replace(np.inf, 1, inplace = True)
    new_data.replace(-np.inf, 0, inplace = True)
    new_data = new_data.fillna(0)
    
    # check missing value or error
    print('# NaN fields: {0}'.format(new_data.isna().sum().sum()))
    print('Max: {0}'.format(new_data.max().max()))
    print('Min: {0}'.format(new_data.min().min()))
    
    array_x = np.array([]).reshape(len(input_col_list),-1,input_period)
    array_y_max = np.array([])
    array_y_min = np.array([])
    array_y_gap_opp = np.array([])
    final_x = np.array([]).reshape(-1,1,input_period)
    
    if for_train == True:
        
        for i in range(data.shape[0] - input_period - output_period_max + 1):
            # reset every row
            temp_x = np.array([]).reshape(-1,1,input_period)
            for j in input_col_list:
                temp_x = np.concatenate((temp_x, 
                                         np.array(new_data[j][i:i+input_period]).reshape(-1,1,input_period))).reshape(-1,1,input_period)
            
            array_x = np.append(array_x, temp_x, axis = 1)
            
            array_y_max = np.append(array_y_max, 
                                    (np.max(np.array(data['High'][i+input_period:i+input_period+output_period_max])) - 
                                     np.array(data['Close'][i+input_period-1]))/np.array(data['Close'][i+input_period-1]))
            
            array_y_min = np.append(array_y_min, 
                                    (np.min(np.array(data['Low'][i+input_period:i+input_period+output_period_min])) - 
                                     np.array(data['Close'][i+input_period-1]))/np.array(data['Close'][i+input_period-1]))
            
            array_y_gap_opp = np.append(array_y_gap_opp, 
                                        (np.max(np.array(data['High'][i+input_period:i+input_period+output_period_max])) - 
                                         np.min(np.array(data['Low'][i+input_period:i+input_period+output_period_min])))/
                                         np.min(np.array(data['Low'][i+input_period:i+input_period+output_period_min])))
            
        assert array_y_max.shape == array_y_min.shape == array_y_gap_opp.shape
        
        # for latest prediction
        for j in input_col_list:
            final_x = np.concatenate((final_x, np.array(new_data[j][-input_period:]).reshape(-1,1,input_period))).reshape(-1,1,input_period)
            
        # add back to main array for any further processing
        array_x = np.append(array_x, final_x, axis = 1)
        
        # order: sample, steps, features
        array_x = np.swapaxes(np.swapaxes(array_x, 0, 1), 1, 2)
            
        print('Input X shape (+1): {0}'.format(array_x.shape))
        print('Input y shape: {0}'.format(array_y_max.shape))
        
        return array_x, array_y_max, array_y_min, array_y_gap_opp

    # for transform purpose only
    else:
        for i in range(data.shape[0] - input_period + 1):
            # reset every row
            temp_x = np.array([]).reshape(-1,1,input_period)
            for j in input_col_list:
                temp_x = np.concatenate((temp_x, 
                                         np.array(new_data[j][i:i+input_period]).reshape(-1,1,input_period))).reshape(-1,1,input_period)
            
            array_x = np.append(array_x, temp_x, axis = 1)
        
        for i in range(data.shape[0] - output_period_max):
            array_y_max = np.append(array_y_max, 
                                    (np.max(np.array(data['High'][i+1:i+1+output_period_max])) - 
                                     np.array(data['Close'][i]))/np.array(data['Close'][i]))
            
            
        array_x = np.swapaxes(np.swapaxes(array_x, 0, 1), 1, 2)
        print('Transformed X shape: {0}'.format(array_x.shape))
        print('Transformed Y shape: {0}'.format(array_y_max.shape))
        
        return array_x, array_y_max
    
if purpose == 'train':         
    X, y_max, _, _ = data_transformer(stock, df_combined, input_col_list, input_period, output_period_max, output_period_min)
    
    print('Baseline MAE: {0}'.format(abs(y_max - y_max.mean()).mean()))
    
    def normalize(train):
        train_min = train.min(axis=(0, 1), keepdims=True)
        train_max = train.max(axis=(0, 1), keepdims=True)
        train_norm = train - train_min / (train_max - train_min)
        return train_norm
    
    X_for_model = X[:-1]
    X_for_pred = X[-1:]
    
    def splitData(X,Y,rate,final_test):
        rate = 1-rate
        X_test = X[-final_test:]
        Y_test = Y[-final_test:]
        X_val = X[int(X.shape[0]*rate):-final_test]
        Y_val = Y[int(Y.shape[0]*rate):-final_test]
        X_train = X[:int(X.shape[0]*rate)]
        Y_train = Y[:int(Y.shape[0]*rate)]
        return X_train, Y_train, X_val, Y_val, X_test, Y_test
    
    # Make sure val data from final records
    X_train, Y_train, X_val, Y_val, X_test, Y_test = splitData(X_for_model, y_max, val_size, test_size)
    
    def shuffle(X,Y):
        np.random.seed(10)
        randomList = np.arange(X.shape[0])
        np.random.shuffle(randomList)
        return X[randomList], Y[randomList]
    
    # shuffle to remove date order for train
    X_train, Y_train = shuffle(X_train, Y_train)
    
    
    
    def buildOneToOneModel(shape):
      model = Sequential()
      #model.add(BatchNormalization())
      #512
      model.add(LSTM(256, input_length=shape[1], input_dim=shape[2], 
                     return_sequences=False))     
      model.add(Dropout(0.5)) 
      #32
      #model.add(LSTM(32, return_sequences=False))
      #model.add(Dropout(0.5)) 
      model.add(Activation('relu')) 
      #model.add(LeakyReLU(alpha=0.1))
      # output shape: (1, 1)
      model.add(Dense(1))    # or use model.add(Dense(1))
      #model.compile(loss="mean_squared_error", optimizer="adam")
      model.compile(loss=customLoss, optimizer="adam")
      model.summary()
      return model
    
    model = buildOneToOneModel(X_train.shape)
    
    callback = EarlyStopping(monitor="val_loss", patience=5, verbose=1, mode="auto")
    
    model.fit(X_train, Y_train, epochs=100, batch_size=64, validation_data = (X_val, Y_val), callbacks=[callback])

    model.save(save_dir + '/' + ticker + '/' + 'model_' + str(dateTimeObj) + '.h5')
    
    print('---KPIs---')
    print('Train: {0}'.format(model.evaluate(X_train, Y_train)))
    print('Val: {0}'.format(model.evaluate(X_val, Y_val)))
    print('Test: {0}'.format(model.evaluate(X_test, Y_test)))
    
    print('---Predictions---')
    print("Prediction on next {0} day's high value: ${1: .2f} / {2: .1f}%".format(output_period_max, 
          float(df_combined['Close'][-1]*(model.predict(X_for_pred)+1)), float(model.predict(X_for_pred)*100)))



else: # for prediction
    #keras.objectives.customLoss = customLoss
    print('Loaded model')
    model = load_model(save_dir + '/' + ticker + '/' + 'model_' + str(dateTimeObj) + '.h5',
                       custom_objects={'customLoss': customLoss})
    
full_X, full_y = data_transformer(stock, df_combined, input_col_list, input_period, output_period_max, output_period_min, False)

y_pred = model.predict(full_X)

final_df = df_combined.copy()
final_df['Actual_High_Next_' + str(output_period_max) + '_P'] = np.concatenate((full_y, [None]*(output_period_max))).astype(float)*100
final_df['Predicted_High_Next_' + str(output_period_max) + '_P'] = np.concatenate(([None]*(input_period-1), y_pred.reshape(-1,))).astype(float)*100

print('---Correlations---')
print(final_df[['Actual_High_Next_' + str(output_period_max) + '_P', 'Predicted_High_Next_' + str(output_period_max) + '_P']].dropna(how = 'any').corr())


# round to 1dp
final_df['Actual_High_Next_' + str(output_period_max) + '_P'] = final_df['Actual_High_Next_' + str(output_period_max) + '_P'].round(1)
final_df['Predicted_High_Next_' + str(output_period_max) + '_P'] = final_df['Predicted_High_Next_' + str(output_period_max) + '_P'].round(1)
final_df['MAE_P'] = (final_df['Predicted_High_Next_' + str(output_period_max) + '_P'] - final_df['Actual_High_Next_' + str(output_period_max) + '_P']).abs()

print('% Predict < Actual: {0}'.format((final_df['Predicted_High_Next_' + str(output_period_max) + '_P'] < final_df['Actual_High_Next_' + str(output_period_max) + '_P']).mean()))

# add type
index_list = [i for i in final_df.index if i >= test_live]
if index_list == []:
    get_row = final_df.shape[0] - output_period_max
else:
    get_row = final_df.index.get_loc(min(index_list)) - output_period_max   # live day minus periods used to generate target
type_list = ['Train / Val'] * (get_row - test_size) + ['Test'] * test_size + ['Live'] * (final_df.shape[0] - get_row)

final_df['Type'] = type_list

# for result
final_df.to_csv(save_dir + '/' + ticker + '/' + 'result_' + str(dateTimeObj) + '.csv')

# for KPI
if not os.path.exists(save_dir + '/' + ticker + '/' + 'KPIs' + '.csv'):
    create_file = pd.DataFrame(columns = ['Model', 'Live', 'Test', 'Train / Val', 'Remark'])
    create_file.to_csv(save_dir + '/' + ticker + '/' + 'KPIs' + '.csv', index = None)
    
if clear_kpi == False:
    existing_kpi = pd.read_csv(save_dir + '/' + ticker + '/' + 'KPIs' + '.csv')
else:
    existing_kpi = pd.DataFrame(columns = ['Model', 'Live', 'Test', 'Train / Val', 'Remark'])

add_kpi = final_df.groupby(['Type']).agg({'MAE_P':'mean'}).T
add_kpi['Model'] = 'model_' + str(dateTimeObj) + '.h5'
print('---This model KPI---')
print(add_kpi)
if existing_kpi['Model'].str.contains('model_' + str(dateTimeObj) + '.h5').sum() >= 1:   # replace for predicting with old model
    add_kpi['Remark'] = existing_kpi[existing_kpi['Model'] == 'model_' + str(dateTimeObj) + '.h5']['Remark'].values
else:
    add_kpi['Remark'] = model_remark

add_kpi = add_kpi[['Model', 'Live', 'Test', 'Train / Val', 'Remark']]

combined_kpi = existing_kpi.append(add_kpi)
combined_kpi.drop_duplicates(subset = ['Model'], keep = 'last', inplace = True)
combined_kpi.to_csv(save_dir + '/' + ticker + '/' + 'KPIs' + '.csv', index = None)
    


