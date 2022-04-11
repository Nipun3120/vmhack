import pandas as pd
from fastapi import FastAPI, Request
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import tensorflow as tf

import nest_asyncio
nest_asyncio.apply()

from datetime import datetime, timedelta
from helper import *


week_tweets = pd.DataFrame()

@app.post("/")
async def read_root(request: Request):
    # city, keyword
    data = await request.json()
    keyword = data['keyword'] 
    city = data['city']
    today = datetime.today().date()
    yesterday = today - timedelta(days=1)
    today_60days = today - timedelta(days=60)
    yesterday_60days = yesterday - timedelta(days=60)
    today_80days = today - timedelta(days=90)
    
    stock_60_df = await fetchStockPrices(today, today_80days, keyword)
    stock_yest_60_df = await fetchStockPrices(yesterday, today_80days, keyword)
     
    scaler, model = getRequiredScaler(keyword)

    global week_tweets
    week_tweets, pearson = await getTweetsWithSentiments(city, keyword, today-timedelta(days=10),today)        #check placement of this code

    
    sentiment_mean_today = week_tweets[week_tweets['date'] == str(today)]['sentiment'].mean()
    sentiment_mean_yest =  week_tweets[week_tweets['date'] == str(today-timedelta(days=1))]['sentiment'].mean()

    feature_array = np.append(scaler.fit_transform(np.array(stock_60_df['Close']).reshape(-1, 1)).reshape(1,-1)[0], sentiment_mean_today)       #stop this from calling twice
    yest_feature_array = np.append(scaler.fit_transform(np.array(stock_yest_60_df['Close']).reshape(-1, 1)).reshape(1,-1)[0], sentiment_mean_yest)  
    
    prediction = model.predict(feature_array.reshape(1,61,1))
    prediction_yest = model.predict(yest_feature_array.reshape(1,61,1))
    
    pred_inv_scaled = scaler.inverse_transform(prediction)
    pred_yest_inv_scaled = scaler.inverse_transform(prediction_yest)
    

    top = week_tweets[week_tweets ['date']== str(datetime.today().date())][['tweet', 'sentiment']].values.tolist()[:5]
    bottom = week_tweets[week_tweets['date'] == str(datetime.today().date())][['tweet', 'sentiment']].values.tolist()[-5:]

    corr, p_value = pearson
    return {"prediction":float((pred_yest_inv_scaled - pred_inv_scaled)[0][0]), "top": top, "bottom": bottom, "corr": corr, "p_value": p_value}