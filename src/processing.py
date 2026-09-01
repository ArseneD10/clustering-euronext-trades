import polars as pl
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from datetime import timedelta

import os
from typing import List
import json
from tqdm import tqdm
import requests

def load_market_cap(url):
    r = requests.get(url)
    r.raise_for_status()
    return pl.DataFrame(r.json())

def get_trades_tokenizer(df: pl.DataFrame, cat_columns = ["TradeType", "MifidInstrumentID", "MmtMarketMechanism", "MmtTradingMode", "MmtAlgorithmicIndicator", "Venue"]):
    trade_tokenizer = {}
    for col in cat_columns:
        keys = df.select(col).unique().to_numpy().flatten().tolist()
        tokeniz = {k: v for v, k in enumerate(keys)}
        trade_tokenizer[col] = tokeniz
    return trade_tokenizer


def get_dataset(csv_path: str):
    data = pl.scan_csv(csv_path, schema_overrides={
                    "MmtTradingMode": pl.String, 
                    "MmtMarketMechanism": pl.String,
                    "MifidClearingFlag": pl.String, 
                    "MmtAlgorithmicIndicator": pl.String
                }).filter(pl.col("MifidInstrumentID").str.starts_with("FR")).collect()
    return data

def get_all_dataset(dir: str= "equities_trade"):
    csv_path = [p for p in os.listdir(dir) if p.endswith('.csv')]
    flag = True
    print(f"Found {len(csv_path)} files in directory")
    for path in csv_path:
        if flag:
            df = get_dataset(os.path.join(dir, path))
            flag = False
        else:
            tmp = get_dataset(os.path.join(dir, path))
            df = pl.concat([df, tmp])
    return df

def save_tokenizer(tokenizer: dict, path="tokenizer.json"):
    with open(path, "w") as f:
        json.dump(tokenizer, f)

def load_tokenizer(path="tokenizer.json"):
    return json.load(open(path, "r"))

def encode_df_feature(df: pl.DataFrame, tokenizer: dict):
    cols = list(tokenizer.keys())
    df = df.with_columns(
        [pl.col(col).replace(tokenizer[col]).cast(pl.Int32) for col in cols]
    )
    return df

def compute_df_feature(df: pl.DataFrame, norm_log_return_window: int = 10, norm_volume_window: int = 10):
    MICRO_SECONDS_IN_HOUR = 3600 * 1e6
    PARIS_TRADING_START = 9
    PARIS_TRADING_END = 17.5
    pulsation = 2.0 * np.pi / ((PARIS_TRADING_END - PARIS_TRADING_START)*MICRO_SECONDS_IN_HOUR)

    max_cut = max(norm_log_return_window, norm_volume_window)
    
    d = df.with_columns(
        pl.col('EventTime').str.to_datetime(format= "%Y-%m-%dT%H:%M:%S%.fZ").dt.timestamp().alias("event_ts"),
        pl.col('TradingDateTime').str.to_datetime(format= "%Y-%m-%dT%H:%M:%S%.fZ").dt.timestamp().alias("trading_ts"),
        pl.col('PublicationDateTime').str.to_datetime(format= "%Y-%m-%dT%H:%M:%S%.fZ").dt.timestamp().alias("publication_ts"),
    )
    d = d.sort("EventTime")
    d = d.with_columns([
       
        (pl.col("event_ts") - pl.col("event_ts").min().over("MifidInstrumentID")).alias("rel_time")
    ]).with_columns([
        (pl.col("rel_time") * pulsation).sin().alias("time_sin"),
        (pl.col("rel_time") * pulsation).cos().alias("time_cos")
    ])

    d = d.group_by("MifidInstrumentID").agg([
        pl.col("event_ts").log().diff().alias("delta_ts"),
        pl.col("MifidPrice").log().diff().alias("log_return"), # Further implementation
        pl.col('MifidQuantity').log1p().alias("log_volume"),
        (pl.col("MifidPrice").log().diff() - pl.col("MifidPrice").log().diff().rolling_mean(norm_log_return_window)).alias('norm_log_return'),
        (pl.col("MifidQuantity").log1p() - pl.col("MifidQuantity").log1p().rolling_mean(norm_log_return_window)).alias('norm_volume'),
        pl.col(pl.Int32),

        pl.col(["event_ts", "time_cos", "time_sin", "MifidPrice"])
    ]).with_columns(
        pl.col(pl.List).list.slice(max_cut) # drop nulls
    )


    return d


def get_volatility_seq(data, categorical_col, continous_col, sq_size): 
    TRADING_YEAR_IN_US = 252 * 8.5 * 3600 * 10e6 # day in trading year * hour in trading day
    INF = 1e12
    out = []
    target_log = []
    for k in tqdm(range(len(data))):
        row = data[k]
        dt = (
            row.explode(pl.all().exclude([pl.String, pl.Float64, pl.Int32, pl.Int64]))
                .drop_nulls()
                .drop_nans()
            )

        length = len(dt)
        if length < sq_size or dt.is_empty():
            continue

        arr   = dt.sort("event_ts").select(continous_col).to_numpy()        
        cat  = dt.sort("event_ts").select(categorical_col).to_numpy()

        TIME_IN_US = (dt.select("event_ts").max() - dt.select("event_ts").min()).item() / len(dt)
        annualized_term = np.sqrt(TRADING_YEAR_IN_US / TIME_IN_US) if TIME_IN_US != 0 else INF
        
        #window_size = length // sq_size
  
        delta = dt.sort('event_ts').select("log_return").to_numpy().flatten() #.reshape(-1, 1) | for array split
        
        delta_window = sliding_window_view(delta, window_shape=sq_size) # np.array_split(delta, window_size)
        windows = sliding_window_view(arr, window_shape=(sq_size, arr.shape[-1])) # np.array_split(arr, window_size) 
        cat_windows = sliding_window_view(cat, window_shape=(sq_size, cat.shape[-1])) # np.array_split(cat, window_size)
        
        out.extend(
            {  
                "categorical": cat_windows[i][0].copy().astype(np.float32),
                "data": windows[i][0].copy().astype(np.float32),
                "target": np.array(
                    delta_window[i].std() * annualized_term if (delta_window[i] != 0).sum() != 0.0 else 0.0,
                    dtype=np.float32
                )
            }
            for i in range(len(windows))
        )
    target_log = np.array([o["target"] for o in out]) 
    quantiles = [0.25, 0.50, 0.75, 0.9]
    target_quantiles = np.quantile(target_log, q=quantiles)
    print("Target mean --> ", target_log.mean(), "+/-", target_log.std())
    print("references quantile -->", quantiles)
    print("Target quantile -->", target_quantiles)
    return out

def get_sequence_by_isin(data: pl.DataFrame, fun, **fun_args):
    seq = []
    isins = data["MifidInstrumentID"].unique()
    for isin in isins:
        filtered = data.filter(pl.col("MifidInstrumentID") == isin)
        seq.extend(
            fun(data=filtered, **fun_args)
        )

    return seq

def get_vae_seq(data, categorical_col, continous_col, sq_size):  
    out = []
    for k in tqdm(range(len(data))):
        row = data[k]
        
        dt = (
            row.explode(pl.all().exclude([pl.String, pl.Float64, pl.Int32, pl.Int64]))
                .drop_nulls()
                .drop_nans()
            )

        length = len(dt)
        if length < sq_size or dt.is_empty():
            continue

        arr   = dt.sort("event_ts").select(continous_col).to_numpy()        
        cat  = dt.sort("event_ts").select(categorical_col).to_numpy()
        
        windows = sliding_window_view(arr, window_shape=(sq_size, arr.shape[-1])) 
        cat_windows = sliding_window_view(cat, window_shape=(sq_size, cat.shape[-1]))
        
        out.extend(
            {  
                "categorical": cat_windows[i][0].copy().astype(np.int32),
                "data": windows[i][0].copy().astype(np.float32),
            }
            for i in range(len(windows))
        )

    return out

def get_vae_seq_with_lr(data, categorical_col, continous_col, sq_size):  
    out = []
    for k in tqdm(range(len(data))):
        row = data[k]
        
        dt = (
            row.explode(pl.all().exclude([pl.String, pl.Float64, pl.Int32, pl.Int64]))
                .drop_nulls()
                .drop_nans()
            )

        length = len(dt)
        if length < sq_size or dt.is_empty():
            continue

        dt = dt.sort("event_ts")

        arr   = dt.select(continous_col).to_numpy()        
        cat  = dt.select(categorical_col).to_numpy()
        lr = dt.select("log_return").to_numpy()
        
        windows = sliding_window_view(arr, window_shape=(sq_size, arr.shape[-1])) 
        cat_windows = sliding_window_view(cat, window_shape=(sq_size, cat.shape[-1]))
        lr = sliding_window_view(lr, window_shape=(sq_size, lr.shape[-1]))
        
        out.extend(
            {  
                "categorical": cat_windows[i][0].copy().astype(np.int32),
                "data": windows[i][0].copy().astype(np.float32),
                "log_return": lr[i].copy().astype(np.float32).flatten() 
            }
            for i in range(len(windows))
        )

    return out

def get_vae_seq_with_var(data, categorical_col, continous_col, sq_size):  
    out = []
    for k in tqdm(range(len(data))):
        row = data[k]
        
        dt = (
            row.explode(pl.all().exclude([pl.String, pl.Float64, pl.Int32, pl.Int64]))
                .drop_nulls()
                .drop_nans()
            )

        length = len(dt)
        if length < sq_size or dt.is_empty():
            continue

        dt = dt.sort("event_ts")

        arr   = dt.select(continous_col).to_numpy()        
        cat  = dt.select(categorical_col).to_numpy()
        price = dt.select("MifidPrice").to_numpy()
        
        windows = sliding_window_view(arr, window_shape=(sq_size, arr.shape[-1])) 
        cat_windows = sliding_window_view(cat, window_shape=(sq_size, cat.shape[-1]))
        price = sliding_window_view(price, window_shape=(sq_size, price.shape[-1]))
        
        out.extend(
            {  
                "categorical": cat_windows[i][0].copy().astype(np.int32),
                "data": windows[i][0].copy().astype(np.float32),
                "var": np.log(price[i].max() / price[i].min()).copy().astype(np.float32).flatten() 
            }
            for i in range(len(windows))
        )

    return out

def get_volatility_discretize_seq(data, categorical_col, continous_col, sq_size, n_states=3, fit_period = 0.7): 
    # Note : discretizer la volatilité comme ça implique un biais look-ahead, une meilleure approche serait une expanding window
    out = []
    quantile = [n / n_states for n in range(1, n_states)]
    for k in tqdm(range(len(data))):
        row = data[k]
        dt = (
            row.explode(pl.all().exclude([pl.String, pl.Float64, pl.Int32, pl.Int64]))
                .drop_nulls()
                .drop_nans()
            )

        length = len(dt)
        fit_size = int(len(dt)*fit_period)
        if fit_size < sq_size or dt.is_empty():
            continue

        arr   = dt.sort("event_ts").select(continous_col).to_numpy()        
        cat  = dt.sort("event_ts").select(categorical_col).to_numpy()
        
        delta = dt.sort('event_ts').select("log_return").to_numpy().flatten() 
        
        delta_window = sliding_window_view(delta, window_shape=sq_size) 
        windows = sliding_window_view(arr, window_shape=(sq_size, arr.shape[-1]))
        cat_windows = sliding_window_view(cat, window_shape=(sq_size, cat.shape[-1])) 

        rolling_std = np.array([delta_window[i].std() for i in range(len(windows)) if delta_window[i].sum() != 0.], dtype=np.float32)[: int(len(windows)*fit_period)]
        bins = np.quantile(rolling_std, q=quantile)
   
        out.extend(
            {  
                "categorical": cat_windows[i][0].copy().astype(np.int32),
                "data": windows[i][0].copy().astype(np.float32),
                "target": np.digitize(
                    delta_window[i+1].std() if (delta_window[i+1] != 0.0).sum() != 0.0 else 0.0,
                    bins=bins
                )
            }
            for i in range(len(windows)-1)
        )

    return out

def get_cross_volatility_sequence():
    pass

def get_dataset_volatility(df: pl.DataFrame, tokenizer):
        
    d = df.with_columns(
        pl.col('EventTime').str.to_datetime(format= "%Y-%m-%dT%H:%M:%S%.9fZ").dt.timestamp().alias("event_ts"),
        pl.col("MifidPrice").log().diff().ewm_std(span=20).over("MifidInstrumentID").alias("Volatility20"),
        pl.col("MifidPrice").log().diff().ewm_std(span=40).over("MifidInstrumentID").alias("Volatility40"),
        pl.col("MifidPrice").log().diff().ewm_std(span=80).over("MifidInstrumentID").alias("Volatility80"),
        pl.col("MifidPrice").log().diff().over("MifidInstrumentID").alias("log_return"),
        pl.col("MifidInstrumentID").replace(tokenizer["MifidInstrumentID"]).cast(pl.Int32).alias("MifidInstrumentID")

    ).select(["Volatility20", "Volatility40", "Volatility80", "log_return", "MifidInstrumentID", "event_ts"])

    return d

def get_cross_volatility_dataset(df: pl.DataFrame):

    START_HOUR = 9
    START_MIN = 0

    d = (df.with_columns(
        pl.col('EventTime').str.to_datetime(format= "%Y-%m-%dT%H:%M:%S%.9fZ").dt.timestamp().alias("event_ts"),
        pl.col('EventTime').str.to_datetime(format= "%Y-%m-%dT%H:%M:%S%.9fZ").dt.convert_time_zone("Europe/Paris").alias("event_time"),
        pl.col("MifidPrice").log().diff().over("MifidInstrumentID").alias("log_return"),

    )
    .with_columns(
        (pl.col("event_time") - (pl.col("event_time").dt.truncate("1d") + timedelta(days=0, hours=START_HOUR,minutes=START_MIN))).dt.total_milliseconds().log1p().alias("log_duration_since_opening"),
        (pl.col("event_ts").log().diff().alias("delta_ts")),
    )).sort(by="event_time")

    print(d.tail(1))

    d = d.group_by_dynamic("event_time", every="1d", group_by="MifidInstrumentID").agg([
        (pl.col("log_return").std() * 252**0.5).alias("daily_volatility"),
        pl.col("MifidQuantity").log().alias("log_volume"),
        (pl.col("MifidQuantity") / pl.col("MifidQuantity").rolling_mean(window_size=10)).alias("norm_volume"),
        pl.col(pl.Int32),
    ]).sort(by="event_time")

    
    return d

def get_tick_bar(df: pl.DataFrame, tick_size=5) -> pl.DataFrame:
    # -> Utiliser un encodage one hot pour les features catégorique ? ou prendre la catégorie avec + de count ?
    # @todo
    pass


def get_num_embedding(df: pl.DataFrame, cols: List[str]):
    buffer = 10
    num_embedding = []
    for col in cols:
        n = df.select(col).unique().__len__() + buffer
        num_embedding.append(n)
    return num_embedding

def get_num_embedding_tokenizer(tokenizer: dict, cols: List[str]):
    buffer = 10
    num_embedding = []
    for col in cols:
        n = len(tokenizer[col]) + buffer
        num_embedding.append(n)
    return num_embedding



if __name__ == "__main__":
    save_path = 'tokenizer.json'
    df = get_all_dataset()
    tokenizer = get_trades_tokenizer(df)
    print("Tokenizer->", tokenizer)
    #save_tokenizer(tokenizer, path=save_path)
    df = encode_df_feature(df, tokenizer=tokenizer)
    #print(df.tail())

    #df = compute_df_feature(df)
    #print(df.tail())

    #ccol = ["delta_event_ts", "log_return", "log_volume", "norm_log_return", "norm_volume"]
    #seq = get_volatility_seq(df, list(tokenizer.keys()), continous_col=ccol, sq_size=20)

    #print(seq[-2])

    vol_df = get_cross_volatility_dataset(df)
    print(vol_df)