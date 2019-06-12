import pandas as pd
import numpy as np

class BBS:
    """ Block bootstrap irregularly-spaced time-series data
    """

    def __init__(self, timeseries, block_length=100, freq='D'):
        """
        Parameters
        ----------
        timeseries : pandas.Series or DataFrame
            Time-indexed data series to be bootstrapped

        block_length : int

        freq : {'D','H','M','S'}
            Frequency. For example D for days
        """
        #assert isinstance(timeseries, pd.Series), ValueError('timeseries must be a pandas Series.')
        self.ts = timeseries.copy()

        assert isinstance(block_length, int), ValueError('block_length must be int')
        self.block_length = pd.Timedelta(f'{block_length} {freq}')

        # prep timeseries
        offset = pd.Timedelta(f'1 {freq}')
        self.ts = self.ts.sort_index()
        self.start = self.ts.index[0] - self.block_length + offset
        self.end = self.ts.index[-1]
        self.ts_range = pd.date_range(self.start, self.end, freq=freq)
        #self.N = self.ts.count()
        self.N = self.ts.shape[0]
        self.N_ts = self.ts_range.shape[0]

    def sample(self, seed=None):
        """ Sample self.ts with replacement until the length of the bootstrap sample
        is equal to the length of self.ts

        Returns
        -------
        A time-indexed data series sampled from the original by block bootstrap.
        """
        N_r = 0
        #values = np.empty(self.N)
        values = np.empty_like(self.ts)
        times = np.empty(self.N, dtype='<M8[ns]')

        while N_r < self.N:
            # establish the time window
            np.random.seed(seed)
            i = np.random.randint(0, self.N_ts)
            start_time = self.ts_range[i]
            end_time = start_time + self.block_length
            #end = start + self.block_length
            start_i = self.ts.index.searchsorted(start_time, side='left')
            end_i = self.ts.index.searchsorted(end_time, side='right')

            # select all sample values from that time window
            block = self.ts[start_i: end_i]
            N_b = block.shape[0] # get length of block
            N_b = min(block.shape[0], self.N - N_r)

            #i_2 = max(-1, self.N - N_r)
            values[N_r:N_r + N_b] = block[0:N_b]
            times[N_r:N_r + N_b] = block.index[0:N_b]
            N_r += N_b

        bootstrap = pd.DataFrame(data=values, index=times, columns=self.ts.columns)
        return bootstrap.sort_index()
