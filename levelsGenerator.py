#       24/12
#       Generate levels for given tickers and write to file for transaq ATF script
#
import json
import os
import coloredlogs
import logging
import sys

import argparse
import constants
from pathlib import Path
import textwrap

#
# finde cluster levels on market data
#

# from tapy import Indicators
from sklearn.cluster import KMeans
import pandas as pd, numpy as np
from datetime import datetime, date, timedelta
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from collections import namedtuple
import mplfinance as mpf

from moex import loadCandlesPage, candles, series, toDfSeries, futSeries, secInfo, validCandlesInterval


def kMeansCentrioids(x):
    """Preprocessing Kmeans. Add Kmeans levels """
    kmeans = KMeans(init="k-means++", n_clusters=20, n_init=4) # , n_clusters=20, n_init=4
    #x = list(df)
    a = kmeans.fit(np.reshape(x,(len(x),1)))

    return  np.sort(np.transpose(kmeans.cluster_centers_))
    labels = kmeans.labels_
    print(np.sort(centroids))
    print(labels)

def normalization(df,columnSuffix = '', period=233, fixScale=0.1, fixShift = 0.5):
    """Preprocessing normalize
    window based mean calculated on Open for [Open, High, Low, Close], fixScale mashtab koefficient and fixShift  applied to fit in range 0..1
    And window base mean, std for Vollume , fixScale mashtab koefficient applied"""
    scale = df['Open'].rolling(window=period).mean()
    for column in df.columns:
        if column in ['Open', 'High', 'Low', 'Close']:
#         df[colName+'_norm'] = preprocessing.normalize(df[colName], norm='l2')
            df[column+columnSuffix] = fixShift + fixScale * (df[column] - scale) / df[column].rolling(window=period).std()
        elif column in ['Volume']:
            df[column+columnSuffix] = fixScale * (df[column] - df[column].rolling(window=period).mean()) / df[column].rolling(window=period).std()
        if any(df[column].loc[(df[column]>=1) & (df[column]<=0)]):
            print(f'Collumn: {column}')

def calculate_fractals(df):
    """Calculate fractals"""
    df['fHigh'] = np.where(
      (df['High'] > df['High'].shift(1)) &
      (df['High'] >= df['High'].shift(-1)) &
      (df['High'] > df['High'].shift(2)) &
      (df['High'] >= df['High'].shift(-2))
      , np.True_, np.False_
      )

    df['fLow'] = np.where(
        (df['Low'] < df['Low'].shift(1)) &
        (df['Low'] <= df['Low'].shift(-1)) &
        (df['Low'] < df['Low'].shift(2)) &
        (df['Low'] <= df['Low'].shift(-2))
        , np.True_, np.False_
    )

# def indicators(df):
#     """Preprocessing add indicators"""
#     i= Indicators(df)
#     # i.fractals(column_name_high='fHigh', column_name_low='fLow')
#     i.sma()
#     i.atr()
#     return  i.df


def calcLevels(df, validateLearnRatio=0.9):
    """ db scan setting for level clusterization
     lernValidateRatio ratio of records out of the forecastiong for validation"""
    epsLev0, epsLev1  = 800, 0.03 # 0.039
    # print(df.shape, df.index[0],df.index[-1])
    dfPast = df.iloc[:int(df.shape[0]*validateLearnRatio)]
    fh = dfPast.loc[dfPast.fHigh==True].High
    fl = dfPast.loc[dfPast.fLow==True].Low
#     print(df.fHigh & (df.index <= df.index.max() - pd.Timedelta(seconds=1)))
#     fig, ax = plt.subplots(figsize=(20,20/1.68)) #,title=tik
    #fig = plt.figure(figsize=(40,15))
    apds, levels  = [],[]
    for j,f in  enumerate([[fh,kMeansCentrioids(list(fh)),'r'],[fl,kMeansCentrioids(list(fl)),'b']]):
        # print(j,f[1])
        levels = levels + list(f[1][0])

        #for i,l in enumerate(f[1][0]):
#         apds = apds + [mpf.make_addplot([l]*len(df), panel='main', color=f[2], linestyle='--') for l in f[1][0]]
        # mpf.make_addplot([2770]*len(df), panel='main', color='r', linestyle='--'),
        # ]
    x1 = np.array(levels).reshape(-1, 1)
    db = DBSCAN(eps=epsLev1, min_samples=1).fit(x1)
    # fractals to plot
    fhToPlot = df.High * df.fHigh #& (df.index <= dfPast.index.max()))
    fhToPlot[fhToPlot==0] = np.nan
    flToPlot = df.Low * df.fLow
    flToPlot[flToPlot==0] = np.nan
    lev = list([np.array(levels)[np.where(db.labels_==k)].mean() for k in np.unique(db.labels_)])
    return lev


def genLevelsForTiker(tikers=[], output_filepath='.//',start='', end='', interval='1', logger=logging.Logger):
    start = start if start else pd.Timestamp.today().floor(freq='D')
    end = end if end else (pd.Timestamp.today() + pd.Timedelta('1D')).floor(freq='D')
    output_filepath = Path(output_filepath)
    assert output_filepath.exists() and output_filepath.is_dir(), f'Wrong path: {output_filepath}'
    for tik in tikers:
        logger.info(f'Tiker: {tik}')
        exportFilePath = output_filepath / (tik + '.txt')
        logger.info(f'Filename: {exportFilePath}')
        tikInfo, engine, market, board, tikInfoTraded = secInfo(tik) # Load tiker info
        if not tikInfo:
            print(f'Error loading {tik}')
            exit(-1)
        # Load data page by pege till got empty data

        df = pd.DataFrame()
        freq = 'W' if interval in ['1'] else 'M'
        for day in pd.period_range(start=start,end=end,freq=freq):
            lastRow = df.shape[0]
            for timeout in range(100):
                dfTmp = candles(sec=tik, interval=interval,
                                dateFrom=day.start_time, #left.to_pydatetime(),
                                dateTo=day.end_time, #right.to_pydatetime(),
                                start=str(df.shape[0]-lastRow)
                                )
                if dfTmp.shape[0] == 0: # chek df is empty then exit
                    break
                df = pd.concat([df,dfTmp])
                logger.debug(f'df.shape{df.shape}, {df.index.max()}')

        # Rename columns
        for i,name in enumerate(['Open', 'Close', 'High', 'Low', 'value', 'Volume']):
            df.rename(columns={ df.columns[i]: name }, inplace = True)

        # Set the combined date-time as the index
        df.index.name = 'datetime'

        # We need df with standart OHLCV columns only
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        logger.info(str(df.shape))
        calculate_fractals(df)
        levels = calcLevels(df, validateLearnRatio=1) # get leves
        try:# ls
            if tikInfo.type  in ['futures_forts','futures']:
                levels = list(map(lambda x: int((x//tikInfoTraded.securities.MINSTEP)*tikInfoTraded.securities.MINSTEP), levels)) # round as ticker price step
            elif tikInfo.type in ['stock_index','stock_index_eq','stock_shares','common_share']:
                levels = list(map(lambda x: round(x, tikInfoTraded.securities.DECIMALS), levels))
        except:
            pass # in INDEX tikers no tikInfo.MINSTEP
        # chose 10 levels nearest to ticker last close value
        close = df.iloc[-1].Close
        levels = sorted(levels, key=lambda x: abs(x - close) / close)[:10]
        levels = sorted(levels) # final sort to prevent

        with exportFilePath.open('wt') as f:
            f.write('\n'.join(map(lambda x: str(x),levels[:10])))
        logger.info(f'{tik} levels: '+ '\n'.join(map(lambda x: str(x),levels)))

if __name__ == "__main__":
    # cProfile.run('main()'#, 'profile_output.txt'
    #              )
    parser = argparse.ArgumentParser(description='''Generate levels for market data 
    and write to transaq atf indicator format see https://github.com/yankoval/transaqAtf.git''',
                                     epilog=textwrap.dedent('''   additional information:
             If you vont to use .json config file put LEVELSGENERATOR=file_path env variable or use -c  
             and create LEVELSGENERATOR.json file with content: 
             { "tikers": "SBRF"{ "interval": "1", // timeframe for candles 1, 10, 60 ,24, 7, 30
                                "timedelta": "-1Y", // pandas Time deltas string
                                }
               "output_filepath" : "output_filepath" 
             } 
         '''))

    # Add arguments for file path, and tiker list
    parser.add_argument('-o', '--output_filepath', dest='output_filepath', type=str,
                        help='Output file path.')
    parser.add_argument('-c', '--config_file', dest='config_file', type=str,
                        help='Output file path.')
    parser.add_argument('-d', '--days', dest='days', default=7, type=int,
                        help='Days to analyze.')
    parser.add_argument('-i', '--intervalCandles', dest='intervalCandles', default='60', type=str,
                        choices=validCandlesInterval,
                        help='Candles interval: 1 - min, 10 - 10 min, 60 - hour, 24 - day, 7 - week, 30 - month ')
    parser.add_argument('-t', '--tikers_list', nargs='+', help='Tikers list. Like: "SBER RIH5"')

    # Parse command line arguments
    args = parser.parse_args()

    # logging
    logger = logging.getLogger(__name__)
    coloredlogs.install(level=logging.DEBUG, logger=logger, isatty=True,
                        fmt="%(asctime)s %(levelname)-8s %(message)s",
                        stream=sys.stderr,
                        datefmt='%Y-%m-%d %H:%M:%S')

    logger.debug("Procesing:" + str(args))
    config = {"tikers":{}}
    # read ini file
    config_file = args.config_file
    if Path(__file__).stem.upper() in os.environ:
        config_file = Path(os.environ[Path(__file__).stem.upper()])
    if config_file:
        with config_file.open(mode='r') as f:
            config['tikers'].update(json.load(f).get('tikers',{}))
        logger.debug(f"ini file loaded. Total keys:{len(config['tikers'].keys())}")
    # update with command line tikers list
    if args.tikers_list:
        for tik in args.tikers_list:
            config['tikers'].update({tik:{"timedelta":f'-{args.days} days','intervalCandles': args.intervalCandles}})

    output_filepath = args.output_filepath if args.output_filepath else config.get('output_filepath','.')
    assert config['tikers'], f'No tikers provided: use conf file or --tikers_list command line option.'
    for tik,param in config['tikers'].items():
        # Call the main function with the parsed arguments
        try:
            logger.info(f'{tik} with param:{param} processing.')
            interval = param.get('intervalCandles','10')
            start = pd.Timestamp(param.get('start')) if param.get('start') \
                else pd.Timestamp.today().floor(freq='D') + pd.Timedelta(param.get('timedelta')).floor(freq='D')
            end = pd.Timestamp(param.get('end')) if param.get('start') \
                else (pd.Timestamp.today() + pd.Timedelta('1D')).floor(freq='D')
            genLevelsForTiker(tikers=[tik], output_filepath = args.output_filepath
                              , start=start, end= end, interval=interval,logger=logger
                              )
        except Exception as e:
            logger.error(e)
    logger.info('Доклад окончил.')