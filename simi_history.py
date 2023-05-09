import json
import requests
import pandas as pd
import time
import numpy as np
import os
import re
from tqdm import tqdm
import datetime

crypto_name = 'BTC'

def cal(x):
    if x>= pd.to_datetime('2013-01-01') and x<= pd.to_datetime('2013-12-31'):
        y = 0
    elif x>= pd.to_datetime('2014-01-01') and x<= pd.to_datetime('2014-12-31'):
        y = 1
    elif x>= pd.to_datetime('2015-01-01') and x<= pd.to_datetime('2016-12-31'):
        y = 2
    elif x>= pd.to_datetime('2017-01-01') and x<= pd.to_datetime('2017-12-31'):
        y = 3
    elif x>= pd.to_datetime('2018-01-01') and x<= pd.to_datetime('2018-12-31'):
        y = 4
    elif x>= pd.to_datetime('2019-01-01') and x<= pd.to_datetime('2019-12-31'):
        y = 5
    elif x>= pd.to_datetime('2020-01-01') and x<= pd.to_datetime('2021-11-30'):
        y = 6
    elif x>= pd.to_datetime('2021-12-01') and x<= pd.to_datetime('2022-12-31'):
        y = 7
    else:
        y = 8
    return y

url_address = [ 'https://api.glassnode.com/v1/metrics/market/price_usd_ohlc']
url_name = ['k_fold']
# insert your API key here
API_KEY = '26BLocpWTcSU7sgqDdKzMHMpJDm'
data_list = []
for num in range(len(url_name)):
    print(num)
    addr = url_address[num]
    name = url_name[num]
    # make API request
    res_addr = requests.get(addr,params={'a': crypto_name, 'api_key': API_KEY})
    # convert to pandas dataframe
    ins = pd.read_json(res_addr.text, convert_dates=['t'])
    #ins.to_csv('test.csv')
    #print(ins['o'])
    ins['date'] =  ins['t']
    ins['value'] =  ins['o']
    ins = ins[['date','value']]
    data_list.append(ins)
result_data = data_list[0][['date']]
for i in range(len(data_list)):
    df = data_list[i]
    result_data = result_data.merge(df,how='left',on='date')
#last_data = result_data[(result_data.date>='2016-01-01') & (result_data.date<='2020-01-01')]
last_data = result_data[(result_data.date>='2013-01-01')]
last_data = last_data.sort_values(by=['date'])
last_data = last_data.reset_index(drop=True)
#print(type(last_data))
date = []
open_p = []
close_p = []
high_p = []
low_p = []
for i in range(len(last_data)):
    date.append(last_data['date'][i])
    open_p.append(last_data['value'][i]['o'])
    close_p.append(last_data['value'][i]['c'])
    high_p.append(last_data['value'][i]['h'])
    low_p.append(last_data['value'][i]['l'])
res_data = pd.DataFrame({'date':date,'open':open_p,'close':close_p,'high':high_p,'low':low_p})
res_data['judge'] = res_data['date'].apply(lambda x:cal(x))
#res_data = res_data[res_data.judge == num_list]
#res_data = res_data[res_data.judge==1]
res_data = res_data.sort_values(by=['date'])
res_data = res_data.reset_index(drop=True)  
from scipy import stats
#只和同一阶段内的数据比较，现在是萧条期，只和历史的萧条期比较
last_7day_data = res_data[-14:]
last_7day_price = list(last_7day_data['close'])
print(last_7day_price)
compare_data_1 = res_data[res_data.judge==2]
compare_data_2 = res_data[res_data.judge==5]
compare_data_1 = compare_data_1.reset_index(drop=True)
compare_data_2 = compare_data_2.reset_index(drop=True)
data_list = []
date_list = []
value = []
for i in range(0,len(compare_data_2)-14):
    ins = list(compare_data_2['close'][i:i+14])
    data_list.append(ins)
    ins_date = list(compare_data_2['date'][i:i+14])
    date_list.append(ins_date)
    p = stats.pearsonr(last_7day_price,ins)
    #print(p)
    value.append(p[0])
maxid = value.index(np.max(value))
simi_data = data_list[maxid]
simi_date = date_list[maxid]
last_7day_data['Open'] = last_7day_data['open']
last_7day_data['Close'] = last_7day_data['close']
last_7day_data['High'] = last_7day_data['high']
last_7day_data['Low'] = last_7day_data['low']
last_7day_data = last_7day_data[['date','Open','Close','High','Low']]
last_7day_data = last_7day_data.set_index(last_7day_data['date'])
filename_1 = 'fig_1.jpg'
start_date = str(np.min(last_7day_data['date']))[0:10]
end_date = str(np.max(last_7day_data['date']))[0:10]
type_bk ='%s - %s BTC OHLC Candles'%(start_date,end_date)
#设置绘制K线的基本参数
import mplfinance as mpf
#def draw_bk(title_name,filename,stock,add_plot):
def draw_bk(title_name,filename,stock):
    ##########################
    # 设置marketcolors
    mc = mpf.make_marketcolors(
        up='red',
        down='green',
        edge='i',
        wick='i',
        volume='in',
        inherit=True)

    # 设置图形风格
    s = mpf.make_mpf_style(
        gridaxis='both',
        gridstyle='-.',
        y_on_right=False,
        marketcolors=mc,
        mavcolors=['yellow','blue'])

    kwargs = dict(
        type='candle',
        mav=(5, 10),
        volume=True,
        title=title_name,
        ylabel='OHLC Candles',
        ylabel_lower='Traded Volume',
        figratio=(25, 10),
        figscale=2
        )
    mpf.plot(stock,
             **kwargs,
             style=s,
             show_nontrading=False,
             #addplot = add_plot,
             savefig=filename
             )

import matplotlib.pyplot as plt
import matplotlib as mpl# 用于设置曲线参数
from cycler import cycler
mc = mpf.make_marketcolors(
    up='red',
    down='green',
    edge='i',
    wick='i',
    volume='in',
    inherit=True)

# 设置图形风格
s = mpf.make_mpf_style(
    gridaxis='both',
    gridstyle='-.',
    y_on_right=False,
    marketcolors=mc,
    #mavcolors=['yellow','blue']
)

kwargs = dict(
    type='candle',
    #mav=(5, 10),
    volume=False,
    title=type_bk,
    ylabel='Price',
    ylabel_lower='Traded Volume',
    figratio=(25, 10),
    figscale=1
    )

#add_plot = mpf.make_addplot(sub_ins[['lowerB','upperB','middleB']])
#draw_bk(type_bk, filename,sub_ins)
mpl.rcParams['axes.prop_cycle'] = cycler(
    color=['dodgerblue','teal'])

# 设置线宽
mpl.rcParams['lines.linewidth'] = 0.5

mpf.plot(last_7day_data,
         **kwargs,
         style=s,
         show_nontrading=False,
         #addplot = add_plot,
         savefig=filename_1
         )
plt.show()
simi_df = res_data[res_data.date>=min_simi_date]
pre_simi_df = simi_df.reset_index(drop=True)
simi_df = pre_simi_df[0:14]
simi_df_pre = pre_simi_df[14:19]

simi_df = simi_df.set_index(simi_df['date'])
filename_2 = 'fig_2.jpg'
start_date_p = str(np.min(simi_df['date']))[0:10]
end_date_p = str(np.max(simi_df['date']))[0:10]
type_bk_1 ='%s - %s BTC OHLC Candles'%(start_date_p,end_date_p)
mc = mpf.make_marketcolors(
    up='red',
    down='green',
    edge='i',
    wick='i',
    volume='in',
    inherit=True)

# 设置图形风格
s = mpf.make_mpf_style(
    gridaxis='both',
    gridstyle='-.',
    y_on_right=False,
    marketcolors=mc,
    #mavcolors=['yellow','blue']
)

kwargs = dict(
    type='candle',
    #mav=(5, 10),
    volume=False,
    title=type_bk_1,
    ylabel='Price',
    ylabel_lower='Traded Volume',
    figratio=(25, 10),
    figscale=1
    )

#add_plot = mpf.make_addplot(sub_ins[['lowerB','upperB','middleB']])
#draw_bk(type_bk, filename,sub_ins)
mpl.rcParams['axes.prop_cycle'] = cycler(
    color=['dodgerblue','teal'])

# 设置线宽
mpl.rcParams['lines.linewidth'] = 0.5
mpf.plot(simi_df,
         **kwargs,
         style=s,
         show_nontrading=False,
         #addplot = add_plot,
         savefig=filename_2
         )
plt.show()
# coding=utf-8
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


def jigsaw(imgs, direction, gap=0):
    imgs = [Image.fromarray(img) for img in imgs]
    w, h = imgs[0].size
    if direction == "horizontal":
        result = Image.new(imgs[0].mode, ((w+gap)*len(imgs)-gap, h))
        for i, img in enumerate(imgs):
            result.paste(img, box=((w+gap)*i, 0))
    elif direction == "vertical":
        result = Image.new(imgs[0].mode, (w, (h+gap)*len(imgs)-gap))
        for i, img in enumerate(imgs):
            result.paste(img, box=(0, (h+gap)*i))
    else:
        raise ValueError("The direction parameter has only two options: horizontal and vertical")
    return np.array(result)


img1 = cv2.imread("fig_1.jpg")
img2 = cv2.imread("fig_2.jpg")
img = jigsaw([img1, img2],direction="vertical")
name = '比特币近14天价格变化历史走势最相似时间' + '.png'
cv2.imwrite(name, img)



import telegram
content = '/root/simi_history/' + name
bot = telegram.Bot(token='6219784883:AAE3YXlXvxNArWJu-0qKpKlhm4KaTSHcqpw')

text = '【历史相似行情提示】：%s-%s比特币价格走势与历史%s-%s比特币价格走势最相似，相似度为%s。'%(start_date,end_date,start_date_p,end_date_p,round(np.max(value),2))

bot.sendDocument(chat_id='-840309715', document=open(name, 'rb'))
bot.sendMessage(chat_id='-840309715', text=text)
