def crosscorr(detax, detay, lag=0, method='pearson'):
    """ Lag-N cross correlation.
        Parameters
        —------—
        lag : int, default 0
        datax, datay : pandas.Series objects of equal length

        Returns
        —------—
        crosscorr : float
        """
    return detax.corrwith(detay.shift(lag), method=method)['score']