#       24/12
#       Generate levels for given tickers and write to file for transaq ATF script
#
import os
import coloredlogs
import logging
import sys
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import argparse
import constants
from pathlib import Path
import textwrap

#
# finde cluster levels on market data
#

from tapy import Indicators
from sklearn.cluster import KMeans
import pandas as pd, numpy as np
import io
import urllib.request
from datetime import datetime, date, timedelta
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from collections import namedtuple
import mplfinance as mpf

from moex import loadCandlesPage, candles, series, toDfSeries, futSeries, secInfo

def kMeansCentrioids(x):
    """Preprocessing Kmeans. Add Kmeans levels """
    kmeans = KMeans(init="k-means++") # , n_clusters=20, n_init=4
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
            print(f'Collumn: {c}')

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

def indicators(df):
    """Preprocessing add indicators"""
    i= Indicators(df)
    # i.fractals(column_name_high='fHigh', column_name_low='fLow')
    i.sma()
    i.atr()
    return  i.df


def plotDay(df, dfS, validateLearnRatio=0.9, plot=True):
    # db scan setting for level clusterization
    # lernValidateRatio ratio of records out of the forecastiong for validation
    epsLev0, epsLev1  = 800, 0.03 # 0.039
    print(df.shape, df.index[0],df.index[-1])
    dfPast = df.iloc[:int(df.shape[0]*validateLearnRatio)]
    fh = dfPast.loc[dfPast.fHigh==True].High
    fl = dfPast.loc[dfPast.fLow==True].Low
#     print(df.fHigh & (df.index <= df.index.max() - pd.Timedelta(seconds=1)))
#     fig, ax = plt.subplots(figsize=(20,20/1.68)) #,title=tik
    #fig = plt.figure(figsize=(40,15))
    apds, levels  = [],[]
    for j,f in  enumerate([[fh,kMeansCentrioids(list(fh)),'r'],[fl,kMeansCentrioids(list(fl)),'b']]):
#         ax.plot(f[0])
#         mpf.plot(df, vollume= True)
#         ax.plot(df.Low)
        print(j,f[1])
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
    if not plot:
        return lev
    apds = apds + [mpf.make_addplot( # add levels to plot
                                    [l]*len(df.iloc[-240:]), panel='main', color='g',label=f'${l:8.0f}$')
                                   for l in lev
          ] + [mpf.make_addplot( # add High fractal points to plot
                            fhToPlot.iloc[-240:]
                            , scatter=True, color='g')
          ] + [mpf.make_addplot( # add Low fractal points to plot
                            flToPlot.iloc[-240:]
                            , scatter=True, color='r')
          ] + [mpf.make_addplot( # add text description of levels to plot
                                     [l]+[np.nan]*(df.iloc[-240:].shape[0]-1), panel='main', type='scatter', marker=f'${l:8.0f}$', markersize=40)
                                  for l in lev
          ]
    print('-----------',len(np.unique(db.labels_)),
         [np.array(levels)[np.where(db.labels_==k)].mean() for k in np.unique(db.labels_)]
         )
    mc = mpf.make_marketcolors(up='g',down='g',
                           edge='lime')
    s  = mpf.make_mpf_style(marketcolors=mc)
    # apds.append(mpf.make_addplot(dfS, title='Real',type='ohlc'))
    # figure, axes =
    mpf.plot(df.iloc[-240:], addplot=apds, volume=True, type='candle', style=s, figscale=10
             ,vlines=dict(vlines=[dfPast.iloc[-240:].index.max(),df.iloc[-240:].index.min()],linewidths=(1,2)),
            fill_between=[
                dict(
                    y1 = df.iloc[-240:]['Close'].shift(+1).values + df.iloc[-240:]['atr'].shift(+1).values*1,
                    y2 = df.iloc[-240:]['Close'].shift(+1).values - df.iloc[-240:]['atr'].shift(+1).values*1,
                    alpha=0.2,color='#291010'),
                dict(
                    y1 = df.iloc[-240:]['Close'].shift(+1).values + df.iloc[-240:]['atr'].shift(+1).values*2,
                    y2 = df.iloc[-240:]['Close'].shift(+1).values - df.iloc[-240:]['atr'].shift(+1).values*2,
                    alpha=0.1,color='#296200'),
                dict(y1 = levels[0],
                     y2 = levels[0] + df.iloc[-240:]['atr'].values,
                     alpha=0.1, color='#296200'),
             ],
            mav=(13, 233),
            returnfig=True
        )
    mpf.show()
    print("------------------------")




def main(tikers=[], output_filepath='.//', days=1, timeFrame='1', logger=logging.Logger):
    for tik in tikers:
        logger.info(f'Tiker: {tik}')
        exportFilePath = output_filepath
        df = pd.DataFrame()

        sInfo = secInfo(tik) # Load tiker info

        if not sInfo['securities']['data']:
            print(f'Error loading {tik}')
            exit(-1)
        s = namedtuple('sec',sInfo['securities']['columns'])
        tikInfo = s(*sInfo['securities']['data'][0])


        # Load data page by pege till got empty data
        for timeout in range(100):
            logger.debug(f'df.shape{df.shape}, {df.index.max()}')
            dfTmp = candles(sec=tik, interval=timeFrame,
                            dateFrom=datetime.now() - timedelta(days=days),
                            dateTo=datetime.now() + timedelta(days=1),
                            start=str(df.shape[0])
                            )
            if dfTmp.shape[0] == 0: # chek df is empty then exit
                break
            df = pd.concat([df,dfTmp])

        # Rename columns
        for i,name in enumerate(['Open', 'Close', 'High', 'Low', 'value', 'Volume']):
            df.rename(columns={ df.columns[i]: name }, inplace = True)

        # Set the combined date-time as the index
        df.index.name = 'datetime'

        # We need df with standart OHLCV columns only
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        df.shape

        dfS = df.copy()
        normalization(df)

        calculate_fractals(df)
        df = indicators(df)
        calculate_fractals(dfS)
        dfS = indicators(dfS)

        levels = plotDay(dfS,df,plot=False)
        levels = list(map(lambda x: int((x // tikInfo.MINSTEP)*tikInfo.MINSTEP), levels))
        with open(exportFilePath + tikInfo.SHORTNAME + '.txt', 'wt') as f:
            f.write('\n'.join(map(lambda x: str(x),levels)))
        print('\n'.join(map(lambda x: str(x),levels)))

if __name__ == "__main__":
    # cProfile.run('main()'#, 'profile_output.txt'
    #              )
    parser = argparse.ArgumentParser(description='Generate levels for market data.',
                                     epilog=textwrap.dedent('''   additional information:
             If you vont to use .ini file put __CONSTANTS__=DEFAULT env variable 
             and create programm_name.ini file with content: 
             [DEFAULT] 
             outFolder = a_section_value
             [tikers]
             tikerList = ['SBER','RIZ4']
         '''))

    # Add arguments for file path, and tiker list
    parser.add_argument('-o', '--output_filepath', dest='output_filepath', default='', type=str,
                        help='Output file path.')
    parser.add_argument('-d', '--days', dest='days', default=7, type=int,
                        help='Days to analyze.')
    parser.add_argument('-t', '--tikers_list', nargs='+', help='Tikers list. <Required> Set flag', required=True)

    # Parse command line arguments
    args = parser.parse_args()

    # logging
    logger = logging.getLogger(__name__)
    coloredlogs.install(level=logging.DEBUG, logger=logger, isatty=True,
                        fmt="%(asctime)s %(levelname)-8s %(message)s",
                        stream=sys.stderr,
                        datefmt='%Y-%m-%d %H:%M:%S')

    logger.debug("Procesing:" + str(args))

    # read ini file
    if '__CONSTANTS__' in os.environ:
        consts = constants.Constants(  # variable='__CONSTANTS__',
            filename=Path(__file__).with_suffix('.ini'))  # doctest: +SKIP
        logger.debug(f'ini:{consts}')
    # Call the main function with the parsed arguments
    main(tikers=args.tikers_list, output_filepath = args.output_filepath, days=args.days, logger=logger)
