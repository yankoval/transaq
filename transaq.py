""" procedures for read transaq export files etc


"""



import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from io import StringIO
import numpy as np
import cProfile

# Your data as a multi-line string
data = """RIU4,25/07/2024 14:41:57,113320,2,К
RIU4,25/07/2024 14:41:58,113330,1,К
RIU4,25/07/2024 14:42:00,113320,1,П
RIU4,25/07/2024 14:42:03,113310,1,П
RIU4,25/07/2024 14:42:03,113320,2,П
CRU4,25/07/2024 09:59:07,11.859,3,К
CRU4,25/07/2024 09:59:07,11.859,2,К
CRU4,25/07/2024 09:59:07,11.859,1,П
CRU4,25/07/2024 09:59:07,11.859,1,К"""


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    col_labels = sorted([x[1] for x in data.columns])
    ax.set_xticks(np.append(np.arange(0,data.shape[1],data.shape[1]//4+1),data.shape[1])
                  , labels=[col_labels[x] for x in np.arange(0,data.shape[1],data.shape[1]//4+1)]+[col_labels[-1]]
                  )
    ax.set_yticks(np.append(np.arange(0, data.shape[0], data.shape[0]//4+1),data.shape[0])
                , labels=[row_labels[x] for x in np.arange(0, data.shape[0], data.shape[0]//4+1)]+[row_labels[-1]]
                )

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    #
    ax.set_xticks(np.arange(0, data.shape[1]+1,50)-.5, minor=True)
    ax.set_yticks(np.arange(0,data.shape[0]+1,50)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            # text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            # texts.append(text)

    return texts


#plot OHLCV market graph
def plot(df,title=''):
    print(mpf.__version__)
    print(mpf.available_styles())
    # Plot the resampled data using mplfinance
    # First we set the kwargs that we will use for all of these examples:
    kwargs = dict(type='candle',
                  # mav=(2, 4, 6),
                  volume=True,
                  figratio=(64, 32), figscale=0.85,
                  style='kenan'
                  # 'kenan', 'mike' ['binance', 'binancedark', 'blueskies', 'brasil', 'charles', 'checkers', 'classic', 'default', 'ibd', 'kenan', 'mike', 'nightclouds', 'sas', 'starsandstripes', 'tradingview', 'yahoo']
                  )
    # Calculate the fractal indicator series
    df['fractal_down'], df['fractal_up'] = add_fractal(df)

    # mpf.plot(df, **kwargs, title=title)
    addl_plots = [mpf.make_addplot(df['fractal_up'],color='#00FF00', panel=0, type='scatter', secondary_y=False),
                  mpf.make_addplot(df['fractal_down'],color='#009F00', panel=0, type='scatter', secondary_y=False)]


    # Plot vertical cumulativ vollume
    bucket_size = 0.0012 * max(df['Close'])
    volprofile = df['Volume'].groupby(df['Close'].apply(lambda x: bucket_size * round(x / bucket_size, 1))).sum()

    fig, axlist = mpf.plot(df,returnfig=True, addplot=addl_plots, **kwargs, title=title)
    vpax = fig.add_axes(axlist[0].get_position())
    vpax.set_axis_off()
    vpax.set_xlim(right=1.2*max(volprofile.values))
    vpax.barh( volprofile.keys().values, volprofile.values, height=0.15*bucket_size, align='center', color='cyan', alpha=0.65)

    mpf.show()

# Define a function to calculate fractal indicators
def add_fractal(df):
    return ( df['Low'][(df['Low'] <= df['Low'].shift(1)) & (df['Low'] <= df['Low'].shift(-1)) & (df['Low'] < df['Low'].shift(2) )& (df['Low'] < df['Low'].shift(-2) )]
            , df['High'][(df['High'] >= df['High'].shift(1)) & (df['High'] >= df['High'].shift(-1)) & (df['High'] > df['High'].shift(2) )& (df['High'] > df['High'].shift(-2) )]
            )

def readTransaqExportFile(file):
    # Read the CSV data into a DataFrame with specified columns and set 'DateTime' as np.datetime
    df = pd.read_csv(file, sep=',', header=None,  names=[
        'ticker', 'datetime', 'price', 'volume', 'Status'], parse_dates=['datetime'],encoding='ansi')

    # Set 'DeviceID' and 'DateTime' as MultiIndex
    df.set_index(['ticker', 'datetime'], inplace=True)
    return df

def TransaqToOHLCV(df):
    gTicTime = df.groupby(['ticker', 'datetime'])
    return pd.DataFrame({'Open': gTicTime.first('price')['price'], 'High': gTicTime.max('price')['price'],
                         'Low': gTicTime.min('price')['price'], 'Close': gTicTime.last('price')['price'],
                         'Volume': gTicTime.sum('volume')['volume']})
def resample(df,tic='',tf='1min'):
    return df.loc[tic].resample('1min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
def main():
    # Use StringIO to simulate a file object from the string data
    # fileName = StringIO(data)
    fileName = r'\\Home-pc\d\share\finam\transaq\240729.txt'
    df = readTransaqExportFile(fileName)
    df_ohlcv = TransaqToOHLCV(df)

    print(df_ohlcv.index.levels[0])
    for tic in df_ohlcv.index.levels[0]:
        resampledDF = resample(df_ohlcv,tic)
        plot(resampledDF, title=tic)
        # plot heat map
        g= df.loc[tic].groupby(['price', pd.Grouper(freq='5min', level=0)])
        df_grouped = g.sum('Price').unstack().sort_index(axis=1, level=1)
        time_idx = sorted([x[1] for x in df_grouped.columns])
        price_idx = df_grouped.index
        fig, ax = plt.subplots()
        im, cbar = heatmap(df_grouped, price_idx, time_idx, ax=ax,
                           cmap="YlGn", cbarlabel="Volume",origin='lower')
        texts = annotate_heatmap(im, valfmt="{x:.1f} t")
        fig.tight_layout()
        plt.title(tic)
        plt.show()
        print(df_grouped)


if __name__ == "__main__":
    # cProfile.run('main()'#, 'profile_output.txt'
    #              )
    main()