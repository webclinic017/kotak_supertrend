from math import floor, ceil
import datetime
import pandas as pd
import numpy as np
import sys
import os
import time
import creds
import ks_api_client
from ks_api_client import ks_api
import pathlib
import requests
from smartapi import SmartConnect
import pandas as pd #to convert data to tables
import urllib.request #to fetch OpenAPIScripMaster


buy_orderslist = []
sell_orderslist=[]
tickerlist = creds.tickerlist
tokenlist = creds.angel_tokenlist
ktokenlist = creds.kotek_tokenlist
Quantity=creds.quantity
supertrend_period = creds.supertrend_period
supertrend_multiplier=creds.supertrend_multiplier
candlesize = creds.candlesize



kotak_ip = '127.0.0.1'
kotak_appId = 'DefaultApplication'
user_id = creds.USER_NAME
user_pwd = creds.PASSWORD
consumer_key = creds.CONSUMER_KEY
consumer_secret = creds.SECRET_KEY
access_token = creds.ACCESS_TOKEN
host = "https://tradeapi.kotaksecurities.com/apim"



import sys
#sys.stdout = open('logfile', 'w')
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
f = open(f"temp/{datetime.datetime.now().strftime('mylogfile_%H_%M_%d_%m_%Y.txt')}", 'w')
backup = sys.stdout
sys.stdout = Tee(sys.stdout, f)





try:
    client = ks_api.KSTradeApi(access_token = access_token, userid = user_id, \
                            consumer_key = consumer_key, ip = kotak_ip, app_id = kotak_appId, \
                            host = host, consumer_secret = consumer_secret)
except Exception as e:
    print("Exception when calling SessionApi->KSTradeApi: %s\n" % e)

try:
    # Login using password
    client.login(password = user_pwd)
except Exception as e:
    print("Exception when calling SessionApi->login: %s\n" % e)

try:
    # Generate final Session Token
    client.session_2fa()
except Exception as e:
    print("Exception when calling SessionApi->session_2fa: %s\n" % e)

print("bot logged in",client)
def second_rounder(t):
    from datetime import datetime, timedelta
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=t.minute, hour=t.hour)
               +timedelta(minutes=1))

def get_angel_historical_data(symboltoken, interval, from_date, end_date):
    ''' 
    symboltoken (str): From the OpenAPIScripMaster.json file
    interval (str): ONE_MINUTE, THREE_MINUTE, FIVE_MINUTE, TEN_MINUTE, FIFTEEN_MINUTE, 
                THIRTY_MINUTE, ONE_HOUR, ONE_DAY
    from_date (str): YYYY-MM-DD HH:MM format
    end_date(str): Defaults to your currentComputer time, unless changed.
    '''
    obj=SmartConnect(api_key="AYq866MN")

    data = obj.generateSession("K433632","Sv@30million")
    
    try:
        historicParam={
        "exchange": "NSE",
        "symboltoken": str(symboltoken),
        "interval": interval,
        "fromdate": from_date, 
        "todate": end_date
        }
        historic_data = obj.getCandleData(historicParam)
    except Exception as e:
        print("Historic Api failed: {}".format(e.message))
    historic_data_df = pd.DataFrame(historic_data['data'])
    historic_data_df.columns=['datetime','open', 'high', 'low', 'close','volume']
    #print("data",historic_data_df)
    return historic_data_df



def saveTokeninfo():
    try:
        path = pathlib.Path(__file__).parent.resolve()
        dff=pd.read_excel(os.path.join(path,"tokens.xlsx"))
        creds.token_info=dff
        print("existing token info imported")
    except:
        print("downloading new data")
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        d = requests.get(url).json()
        token_df = pd.DataFrame.from_dict(d)
        token_df["expiry"] = pd.to_datetime(token_df["expiry"])
        token_df = token_df.astype({"strike":float})
        #token_df=token_df.reset_index(drop=True)
        token_df.to_excel("alltokens.xlsx",index=False)

        df=token_df
        #creds.token_info= df[(df['exch_seg']=="NSE") & (df["instrumenttype"]==("OPTIDX" or "OPTSTK")) & ((df["name"]==("BANKNIFTY")) | (df["name"]==("NIFTY")))]
        creds.token_info= df[(df['exch_seg']=="NSE") ]

        print(creds.token_info)
        creds.token_info.to_excel("tokens.xlsx",index=False)
    df=creds.token_info
    df=df.sort_values(by=['token'])
    for i in creds.tickerlist:
        token=df[(df['name']==str(i))]
        #print("token")
        #print(token["token"].iloc[0])
        creds.angel_tokenlist.append(token["token"].iloc[0])
    
    kotek_token_info()


def kotek_token_info():
    url =  'https://tradeapi.kotaksecurities.com/apim/scripmaster/1.1/filename'
    headers = {'accept' : 'application/json', 'consumerKey' : creds.CONSUMER_KEY, 'Authorization':f'Bearer {creds.ACCESS_TOKEN}'}
    res = requests.get(url,headers=headers).json()
    print(res)
    cashurl = res['Success']['cash']
    fnourl = res['Success']['fno']
    cashdf = pd.read_csv(cashurl,sep='|')
    fnodf = pd.read_csv(fnourl,sep='|')
    
    dfz=cashdf
    creds.index_info= dfz[((dfz["exchange"]==("NSE")))]
    dff=creds.index_info
    for i in creds.tickerlist:
        token=dff[(dff['instrumentName']==str(i))]
        creds.kotek_tokenlist.append(token["instrumentToken"].iloc[0])
    #print("creds token list",creds.kotek_tokenlist)










# Source for tech indicator : https://github.com/arkochhar/Technical-Indicators/blob/master/indicator/indicators.py
def EMA(df, base, target, period, alpha=False):
    """
    Function to compute Exponential Moving Average (EMA)

    Args :
        df : Pandas DataFrame which contains ['timestamp', 'open', 'high', 'low', 'close', 'Volume'] columns
        base : String indicating the column name from which the EMA needs to be computed from
        target : String indicates the column name to which the computed data needs to be stored
        period : Integer indicates the period of computation in terms of number of candles
        alpha : Boolean if True indicates to use the formula for computing EMA using alpha (default is False)

    Returns :
        df : Pandas DataFrame with new column added with name 'target'
    """

    con = pd.concat([df[:period][base].rolling(window=period).mean(), df[period:][base]])

    if (alpha == True):
        # (1 - alpha) * previous_val + alpha * current_val where alpha = 1 / period
        df[target] = con.ewm(alpha=1 / period, adjust=False).mean()
    else:
        # ((current_val - previous_val) * coeff) + previous_val where coeff = 2 / (period + 1)
        df[target] = con.ewm(span=period, adjust=False).mean()

    df[target].fillna(0, inplace=True)
    return df

def ATR(df, period, ohlc=['open', 'high', 'low', 'close']):
    """
    Function to compute Average True Range (ATR)

    Args :
        df : Pandas DataFrame which contains ['timestamp', 'open', 'high', 'low', 'close', 'Volume'] columns
        period : Integer indicates the period of computation in terms of number of candles
        ohlc: List defining OHLC Column names (default ['open', 'high', 'low', 'close'])

    Returns :
        df : Pandas DataFrame with new columns added for
            True Range (TR)
            ATR (ATR_$period)
    """
    atr = 'ATR_' + str(period)

    # Compute true range only if it is not computed and stored earlier in the df
    if not 'TR' in df.columns:
        df['h-l'] = df[ohlc[1]] - df[ohlc[2]]
        df['h-yc'] = abs(df[ohlc[1]] - df[ohlc[3]].shift())
        df['l-yc'] = abs(df[ohlc[2]] - df[ohlc[3]].shift())

        df['TR'] = df[['h-l', 'h-yc', 'l-yc']].max(axis=1)

        df.drop(['h-l', 'h-yc', 'l-yc'], inplace=True, axis=1)

    # Compute EMA of true range using ATR formula after ignoring first row
    EMA(df, 'TR', atr, period, alpha=True)

    return df

def SuperTrend(df, period = supertrend_period, multiplier=supertrend_multiplier, ohlc=['open', 'high', 'low', 'close']):
    """
    Function to compute SuperTrend

    Args :
        df : Pandas DataFrame which contains ['timestamp', 'open', 'high', 'low', 'close', 'Volume'] columns
        period : Integer indicates the period of computation in terms of number of candles
        multiplier : Integer indicates value to multiply the ATR
        ohlc: List defining OHLC Column names (default ['open', 'high', 'low', 'close'])

    Returns :
        df : Pandas DataFrame with new columns added for
            True Range (TR), ATR (ATR_$period)
            SuperTrend (ST_$period_$multiplier)
            SuperTrend Direction (STX_$period_$multiplier)
    """

    ATR(df, period, ohlc=ohlc)
    atr = 'ATR_' + str(period)
    st = 'ST' #+ str(period) + '_' + str(multiplier)
    stx = 'STX' #  + str(period) + '_' + str(multiplier)

    """
    SuperTrend Algorithm :

        BASIC UPPERBAND = (HIGH + LOW) / 2 + Multiplier * ATR
        BASIC LOWERBAND = (HIGH + LOW) / 2 - Multiplier * ATR

        FINAL UPPERBAND = IF( (Current BASICUPPERBAND < Previous FINAL UPPERBAND) or (Previous Close > Previous FINAL UPPERBAND))
                            THEN (Current BASIC UPPERBAND) ELSE Previous FINALUPPERBAND)
        FINAL LOWERBAND = IF( (Current BASIC LOWERBAND > Previous FINAL LOWERBAND) or (Previous Close < Previous FINAL LOWERBAND)) 
                            THEN (Current BASIC LOWERBAND) ELSE Previous FINAL LOWERBAND)

        SUPERTREND = IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close <= Current FINAL UPPERBAND)) THEN
                        Current FINAL UPPERBAND
                    ELSE
                        IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close > Current FINAL UPPERBAND)) THEN
                            Current FINAL LOWERBAND
                        ELSE
                            IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close >= Current FINAL LOWERBAND)) THEN
                                Current FINAL LOWERBAND
                            ELSE
                                IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close < Current FINAL LOWERBAND)) THEN
                                    Current FINAL UPPERBAND
    """

    # Compute basic upper and lower bands
    df['basic_ub'] = (df[ohlc[1]] + df[ohlc[2]]) / 2 + multiplier * df[atr]
    df['basic_lb'] = (df[ohlc[1]] + df[ohlc[2]]) / 2 - multiplier * df[atr]

    # Compute final upper and lower bands
    df['final_ub'] = 0.00
    df['final_lb'] = 0.00
    for i in range(period, len(df)):
        df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or \
                                                         df[ohlc[3]].iat[i - 1] > df['final_ub'].iat[i - 1] else \
        df['final_ub'].iat[i - 1]
        df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or \
                                                         df[ohlc[3]].iat[i - 1] < df['final_lb'].iat[i - 1] else \
        df['final_lb'].iat[i - 1]

    # Set the Supertrend value
    df[st] = 0.00
    for i in range(period, len(df)):
        df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df[ohlc[3]].iat[
            i] <= df['final_ub'].iat[i] else \
            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df[ohlc[3]].iat[i] > \
                                     df['final_ub'].iat[i] else \
                df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df[ohlc[3]].iat[i] >= \
                                         df['final_lb'].iat[i] else \
                    df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df[ohlc[3]].iat[i] < \
                                             df['final_lb'].iat[i] else 0.00

        # Mark the trend direction up/down
    df[stx] = np.where((df[st] > 0.00), np.where((df[ohlc[3]] < df[st]), 'down', 'up'), np.NaN)

    # Remove basic and final bands from the columns
    df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

    df.fillna(0, inplace=True)
    df.to_csv("super.csv")

    return df

def gethistoricaldata(token):

    enddate = datetime.datetime.today()
    startdate = enddate - datetime.timedelta(7)
    enddate=enddate.strftime("%Y-%m-%d %H:%M")
    startdate=startdate.strftime("%Y-%m-%d %H:%M")
    
    
    df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    try:
        #data = yf.download(tickers=token, start=startdate,end=enddate, interval=candlesize)
        #print("getting data from angel")
        dat=get_angel_historical_data(token,candlesize,startdate,enddate)

        #data = kites[0].historical_data(token, startdate, enddate, interval=candlesize)
        df = pd.DataFrame.from_dict(dat, orient='columns', dtype=None)
        #print("da")
        #print(df)
        if not df.empty:
            #df = df[['timestamp', 'open', 'high', 'low', 'close', 'Volume']]
            #df['timestamp'] = df['timestamp'].astype(str).str[:-6]
            #df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = SuperTrend(df)
        df.to_csv('output.csv')
    except Exception as e:
        print("         error in gethistoricaldata", token, e)
    return df


def run_trategy():
    for i in range(0, len(tickerlist)):

        #if (tickerlist[i] in orderslist):
            #continue
        try:
            histdata = gethistoricaldata(tokenlist[i])
            #print(histdata)
            super_trend = histdata.STX.values
            lastclose = histdata.close.values[-1]
            """
            stoploss_buy = histdata.low.values[-3] # third last candle as stoploss
            stoploss_sell = histdata.high.values[-3] # third last candle as stoploss

            if stoploss_buy > lastclose * 0.996:
                stoploss_buy = lastclose * 0.996 # minimum stoploss as 0.4 %

            if stoploss_sell < lastclose * 1.004:
                stoploss_sell = lastclose * 1.004 # minimum stoploss as 0.4 %
            """
            #print("lastclose",lastclose)
            #print("stoploss abs",stoploss)
            print(tickerlist[i],lastclose,super_trend[-4:])
            first_time=int(creds.first_time[i])
            if (first_time==1):
                
                print("opening fresh position")
                quantity = Quantity[i]
                if(super_trend[-1]=='up'):
                    
                    #order = kites[0].place_order(order_type = creds.order_type, instrument_token = int(creds.nifty_ce_token_instrument), transaction_type = "BUY",\
                    #    quantity = creds.quantity, price = 0, disclosed_quantity = 0, trigger_price = 0,\
                    #    tag = "string", validity = "GFD", variety = "REGULAR"
                    #                             )
                    ord_id=client.place_order(order_type = creds.order_type, instrument_token = int(creds.kotek_tokenlist[i]), transaction_type = "BUY",\
                        quantity = creds.quantity[i], price = 0, disclosed_quantity = 0, trigger_price = 0,\
                        tag = "string", validity = "GFD", variety = "REGULAR")
                    print("Order : ",ord_id, "BUY", tickerlist[i], "quantity:",quantity,datetime.datetime.now())
                    buy_orderslist.append(tickerlist[i])
                    creds.first_time[i]=int(0)
                elif(super_trend[-1]=='down'):
                   
                    #
                    #
                    #
                    ord_id=client.place_order(order_type = creds.order_type, instrument_token = int(creds.kotek_tokenlist[i]), transaction_type = "SELL",\
                        quantity = creds.quantity[i], price = 0, disclosed_quantity = 0, trigger_price = 0,\
                        tag = "string", validity = "GFD", variety = "REGULAR")
                    print("Order : ",ord_id, "SELL", tickerlist[i], "quantity:",quantity, datetime.datetime.now())
                    sell_orderslist.append(tickerlist[i])
                    creds.first_time[i]=int(0)




            if super_trend[-1]=='up' :
                if (tickerlist[i] in sell_orderslist):
                    quantity = Quantity[i]
                    print("closing position",tickerlist[i])
                    ord_id=client.place_order(order_type = creds.order_type, instrument_token = int(creds.kotek_tokenlist[i]), transaction_type = "BUY",\
                        quantity = creds.quantity[i], price = 0, disclosed_quantity = 0, trigger_price = 0,\
                        tag = "string", validity = "GFD", variety = "REGULAR")
                    print("Order : ",ord_id, "BUY", tickerlist[i], "quantity:",quantity,datetime.datetime.now())
                    sell_orderslist.remove(tickerlist[i])
                print("sell orders list",sell_orderslist)




                if (tickerlist[i] not in buy_orderslist):
                    quantity = Quantity[i]
                    """
                    stoploss_buy = lastclose - stoploss_buy
                    #print("stoploss delta", stoploss)

                    quantity = floor(max(1, (risk_per_trade/stoploss_buy)))
                    target = stoploss_buy*3 # risk reward as 3

                    price = int(100 * (floor(lastclose / 0.05) * 0.05)) / 100
                    stoploss_buy = int(100 * (floor(stoploss_buy / 0.05) * 0.05)) / 100
                    quantity = int(quantity)
                    target = int(100 * (floor(target / 0.05) * 0.05)) / 100
                    """

                    
                    #order = kites[0].place_order(order_type = creds.order_type, instrument_token = int(creds.nifty_ce_token_instrument), transaction_type = "BUY",\
                    #    quantity = creds.quantity, price = 0, disclosed_quantity = 0, trigger_price = 0,\
                    #    tag = "string", validity = "GFD", variety = "REGULAR"
                    #                             )
                    ord_id=client.place_order(order_type = creds.order_type, instrument_token = int(creds.kotek_tokenlist[i]), transaction_type = "BUY",\
                        quantity = creds.quantity[i], price = 0, disclosed_quantity = 0, trigger_price = 0,\
                        tag = "string", validity = "GFD", variety = "REGULAR")
                    print("Order : ",ord_id, "BUY", tickerlist[i], "quantity:",quantity,datetime.datetime.now())
                    buy_orderslist.append(tickerlist[i])

            if super_trend[-1]=='down' :
                if (tickerlist[i] in buy_orderslist):
                    quantity = Quantity[i]
                    print("closing position",tickerlist[i])
                    ord_id=client.place_order(order_type = creds.order_type, instrument_token = int(creds.kotek_tokenlist[i]), transaction_type = "SELL",\
                        quantity = creds.quantity[i], price = 0, disclosed_quantity = 0, trigger_price = 0,\
                        tag = "string", validity = "GFD", variety = "REGULAR")
                    print("Order : ",ord_id, "SELL", tickerlist[i], "quantity:",quantity, datetime.datetime.now())
                    buy_orderslist.remove(tickerlist[i])
                    print("buy orders list",buy_orderslist)
                
                
                
                if (tickerlist[i] not in sell_orderslist):
                    quantity = Quantity[i]
                    """
                    stoploss_sell= stoploss_sell - lastclose
                    #print("stoploss delta", stoploss)

                    quantity = floor(max(1, (risk_per_trade/stoploss_sell)))
                    target = stoploss_sell*3 # risk reward as 3

                    price = int(100 * (floor(lastclose / 0.05) * 0.05)) / 100
                    stoploss_sell = int(100 * (floor(stoploss_sell / 0.05) * 0.05)) / 100
                    quantity = int(quantity)
                    target = int(100 * (floor(target / 0.05) * 0.05)) / 100
                    """

                    
                    #
                    #
                    #
                    ord_id=client.place_order(order_type = creds.order_type, instrument_token = int(creds.kotek_tokenlist[i]), transaction_type = "SELL",\
                        quantity = creds.quantity[i], price = 0, disclosed_quantity = 0, trigger_price = 0,\
                        tag = "string", validity = "GFD", variety = "REGULAR")
                    print("Order : ",ord_id, "SELL", tickerlist[i], "quantity:",quantity, datetime.datetime.now())
                    sell_orderslist.append(tickerlist[i])

        except Exception as e :
            print(e)
def squareoff():
    
    print("squaring off all open positions")
    for i in range(0, len(tickerlist)):
        if (tickerlist[i] in buy_orderslist):
                        quantity = Quantity[i]
                        print("closing position",tickerlist[i])
                        ord_id=client.place_order(order_type = creds.order_type, instrument_token = int(creds.kotek_tokenlist[i]), transaction_type = "SELL",\
                            quantity = creds.quantity[i], price = 0, disclosed_quantity = 0, trigger_price = 0,\
                            tag = "string", validity = "GFD", variety = "REGULAR")
                        print("Order : ",ord_id, "SELL", tickerlist[i], "quantity:",quantity, datetime.datetime.now())
                        buy_orderslist.remove(tickerlist[i])
                        print("buy orders list",buy_orderslist)
        if (tickerlist[i] in sell_orderslist):
                        quantity = Quantity[i]
                        print("closing position",tickerlist[i])
                        ord_id=client.place_order(order_type = creds.order_type, instrument_token = int(creds.kotek_tokenlist[i]), transaction_type = "BUY",\
                            quantity = creds.quantity[i], price = 0, disclosed_quantity = 0, trigger_price = 0,\
                            tag = "string", validity = "GFD", variety = "REGULAR")
                        print("Order : ",ord_id, "BUY", tickerlist[i], "quantity:",quantity,datetime.datetime.now())
                        sell_orderslist.remove(tickerlist[i])
                        print("sell orders list",sell_orderslist)
    print("squared off all open positions")

def run():
    saveTokeninfo()
    print("user data loaded..........", datetime.datetime.now())
    global runcount
    sts=second_rounder(datetime.datetime.now())
    cur_time = int(sts.hour) * 60 + int(sts.minute)
    start_time = creds.start_time  # specify in int (hr) and int (min) foramte
    end_time = creds.end_time  # do not place fresh order
    #stop_time = int(15) * 60 + int(25)  # square off all open positions
    last_time = start_time
    schedule_interval = creds.schedule  # run at every 1 min
    #runcount = 0
    while True:
        if (datetime.datetime.now().hour * 60 + datetime.datetime.now().minute) >= end_time:
                print(sys._getframe().f_lineno, "Trading day closed, time is above stop_time")
                squareoff()
                break

        if (datetime.datetime.now().hour * 60 + datetime.datetime.now().minute) >= start_time:
            if (datetime.datetime.now().hour * 60 + datetime.datetime.now().minute) >= cur_time:
                if time.time() >= last_time:
                    last_time = time.time() + schedule_interval
                    print("\n\n {} Run Count : Time - {} ".format(runcount, datetime.datetime.now()))
                    if runcount >= 0:
                        try:
                            run_trategy()
                        except Exception as e:
                            print("Run error", e)
                    runcount = runcount + 1
        else:
            print('     Waiting...', datetime.datetime.now())
            time.sleep(1)

runcount = 0
run()

