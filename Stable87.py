import allantools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from matplotlib.gridspec import GridSpec
from datetime import datetime
from dateutil.relativedelta import relativedelta
from lmfit.models import LinearModel


def get_better_ade(tau, adev, tau0, N, alpha=0, d=2, overlapping=True, modified=False):
    """Calculate non-naive Allan deviation errors. Equivalent to Stable32.

    Ref:
        https://github.com/aewallin/allantools/blob/master/examples/ci_demo.py
        https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20050061319.pdf

    Args:
        tau (list of floats):           list of tau_values for which deviations were computed
        adev (list of floats):          list of ADEV (or another statistic) deviations
        tau0 (float):                   averaging factor;  average interval between measurements
        N (int):                        number of frequency observations
        alpha (int, optional):          +2,...,-4   noise type, either estimated or known
        d (int, optional):              statistic code: 1 first-difference variance, 2 allan variance, 3 hadamard
                                        variance
        overlapping (bool, optional):   True if overlapping statistic used. False if standard statistic used
        modified (bool, optional):      True if modified statistic used. False if standard statistic used.

    Returns:
        err_lo (list of floats):        non-naive lower 1-sigma confidence interval for each point over which deviations
                                        were computed
        err_high (list of floats):      non-naive higher 1-sigma confidence interval for each point over which deviations
                                        were computed
    """

    # Confidence-intervals for each (tau, adev) pair separately.
    cis = []
    for (t, dev) in zip(tau, adev):
        # Greenhalls EDF (Equivalent Degrees of Freedom)
        edf = allantools.edf_greenhall(alpha=alpha, d=d, m=t / tau0, N=N, overlapping=overlapping, modified=modified)
        # with the known EDF we get CIs
        (lo, hi) = allantools.confidence_interval(dev=dev, edf=edf)
        cis.append((lo, hi))

    err_lo = np.array([d - ci[0] for (d, ci) in zip(adev, cis)])
    err_hi = np.array([ci[1] - d for (d, ci) in zip(adev, cis)])

    return err_lo, err_hi


def oadev(data, taus, scale=1/429228004229873, alpha=0):
    """Calculate overlapping Allan deviation with non-naive errors, from frequency data file.

    Args:
        data (array):          array of fractional frequency data file whose first column is a timestamp in (s)
        taus (list of float):       list of tau-values for OADEV computation
        scale (float, optional):    scaling factor for fractional frequency. Defaults to 1/Sr87_BIPM_frequency
        alpha (int, optional):      +2,...,-4   noise type, either estimated or known. Defaults to 0 to match Stable 32

    Returns:
        t2 (list of floats):                            list of tau_values for which deviations were computed
        ad_ff (list of floats):                         list of oadev deviations in fractional frequency units
        err_lo_ff (list of floats):                     list of non-naive lower 1-sigma errors for each point over which
                                                        deviations were computed
        err_hi_ff (list of floats):                     list of non-naive higher 1-sigma errors for each point over
                                                        which deviations were computed
        adn (list):                                     list of number of pairs in overlapping allan computation

    """

    x = data[:, 0]  # Get timestamps
    y = data[:, 1]  # Get fractional frequency data

    avg_interval = (x[-1] - x[0]) / len(x)  # average interval between measurements
    r = 1 / avg_interval  # average sample rate in Hz of the input data

    t = np.array(taus) * avg_interval  # tau values on which to evaluate metric
    (t2, ad, ade, adn) = allantools.oadev(y, rate=r, data_type="freq", taus=t)  # normal ODEV computation, giving naive 1/sqrt(N) errors

    # correct for deadtime ad/np.sqrt(B2*B3)
    # https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication1065.pdf | 5.15 Dead Time
    # TODO

    # Get correct (Stable32) errors
    err_lo, err_hi = get_better_ade(t2, ad, avg_interval, len(x), alpha=alpha, d=2, overlapping=True, modified=False)

    ad_ff = ad*scale
    err_lo_ff = err_lo*scale
    err_hi_ff = err_hi*scale

    return t2, ad_ff, err_lo_ff, err_hi_ff, adn


def print_summary(tau, adev, err_lo, err_hi, adn=None, scale=1/429228004229873):
    """Prints summary statistics at each computed tau value.
    """
    # Assumes data in fractional frequency units

    ci_lo = np.array([d - e_lo for (d, e_lo) in zip(adev, err_lo)])
    ci_hi = np.array([e_hi + d for (d, e_hi) in zip(adev, err_hi)])
    cis = np.vstack((ci_lo, ci_hi)).T

    # Print and plot the results
    #print("Tau\tmin Dev\t\tDev\t\tMax Dev")
    #for (t, dev, ci) in zip(tau, adev*10**17, cis*10**17):
    #    print("%d\t%f\t%f\t%f" % (int(round(t)), ci[0], dev, ci[1]))

    # Stable 32 style legend
    legend_text = "Tau\tSigma\n"
    for (t, dev) in zip(tau, adev*10**17): 
        legend_text += "%d\t%.2f\n" % (int(round(t)), dev)

    return legend_text.expandtabs()


def fit_adev(tau, adev, err_lo, err_high):

    fit_tau_over = 499

    # If there are at least 2 datapoints to fit at large tau_values, fit them
    if len(tau[np.where(tau > fit_tau_over)]) >= 2:

        # TODO: take into account asymmetric errorbars
        weights = (err_lo + err_high) / 2  # take naive 1-std errorbar average

        x = np.array([t for t in tau if t > fit_tau_over])  # only fit long tau values
        y = np.array([a for i, a in enumerate(adev) if tau[i] > fit_tau_over])  # take equivalent in adev array
        w = np.array([h for i, h in enumerate(weights) if tau[i] > fit_tau_over])   # take equivalent in weights array

        # Fit straight line on a log10 scale
        x = np.log10(x)
        y = np.log10(y)
        w_log = np.log10(np.exp(1))*(w/y)  # error propagation for log10(y +- w)

        # Weighted Least Squares fit; ax + b
        model = LinearModel()

        params = model.make_params()
        params['intercept'].max = -10
        params['intercept'].value = -15
        params['intercept'].min = -19
        params['intercept'].brute_step = 0.005
        params['slope'].value = -0.5  # assume white noise dominates on fitting range
        params['slope'].vary = False  # ... so we keep this parameter fixed

        res = model.fit(y, params, weights=1/w_log**2, x=x)

        a = res.values['slope']
        b = res.values['intercept']

        x_smooth = np.logspace(0, 5, 20)
        return x_smooth, 10 ** (res.eval(x=np.log10(x_smooth))), a, b

    # Else if there are not enough large tau_values to fit, return empty arrays
    else:

        return [], [], 0.5, -1


def reject_outliers(x, y, m=5.189, switch_polarity=False):
    # https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    # If switch polarity is True, the function will return the outliers

    d = np.abs(y - np.median(y))
    mdev = np.median(d)
    s = np.abs(d/mdev) if mdev else 0.

    if switch_polarity:
        my = np.ma.masked_where(s < m, y)
    else:
        my = np.ma.masked_where(s > m, y)

    mx = np.ma.masked_where(np.ma.getmask(my), x)  # masked array

    return mx[~mx.mask], my[~my.mask]  # get unmasked data


def convert_labView_timestamp(timestamps):

    # LabView epoch starts in 1904 (66 years before UNIX epoch) also notice that 2020 was a leapyear
    return [(datetime.fromtimestamp(timestamp) - relativedelta(years=66, leapdays=-1)).strftime('%Y-%m-%d\n%H:%M:%S') for timestamp in timestamps]


def distinct_ranges(ranges, new_range):
    # merge multiple index selections into distinct unique ranges
    # ranges (list): [[xmin, xmax], ...]
    # new_range (list): [xmin, xmax]

    new_xmin = new_range[0]
    new_xmax = new_range[1]

    ranges = sorted(ranges, key=lambda x: x[0])

    flag_call_again = False
    flag_return = False

    for i, range in enumerate(ranges):

        xmin = range[0]
        xmax = range[1]

        if new_xmin <= xmax and new_xmax >= xmin:
            xmin = min(xmin, new_xmin)
            xmax = max(xmax, new_xmax)
                
            ranges[i] = [xmin, xmax]

            if len(ranges) > 1:
                flag_call_again = True
                break
            else:
                flag_return = True
                break

        else:
            
            if i == len(ranges) - 1:
                ranges.append(new_range)
                flag_return = True
                break

    if flag_call_again:
        ranges = distinct_ranges(ranges[:-1], ranges[-1])

    if flag_return:
        return sorted(ranges, key=lambda x: x[0])  # recursive output

    return ranges  # final output


class Onselect:

    def __init__(self, fullx, fully, fig, ax1, ax2, line2, line2a, ax3, line4, ax4):

        self.fullx = fullx
        self.fully = fully
        self.fig = fig
        self.ax1 = ax1  # master axis on which the span selector lives

        self.ax2 = ax2
        self.line2 = line2
        self.line2a = line2a
        self.ax3 = ax3
        self.line4 = line4
        self.ax4 = ax4

        self.intervals = [[0, 0]]
        self.data_sel = None  # selected data

    def __call__(self, xmin, xmax):

        # On each call:
        # - Slice data according to selections
        # - Update object data with selected data
        # - Update Plots

        indmin, indmax = np.searchsorted(self.fullx, (xmin, xmax))
        indmax = min(len(self.fullx)-1, indmax)

        self.intervals = distinct_ranges(self.intervals, [indmin, indmax])

        # Calulate data to keep for data analysis (here we select with the SpanSelector data we want to remove)
        slices = []
        for ind_range in self.intervals:  # create proper slice objects
            slices.append(slice(ind_range[0], ind_range[1]+1))
        keep_mask = np.ones_like(self.fullx, dtype=bool)
        for sl in slices:
            keep_mask[sl] = False
        x_sel = self.fullx[keep_mask]  # remove slices
        y_sel = self.fully[keep_mask]
        self.data_sel = np.vstack((x_sel, y_sel)).T  # total data selected (i.e. not selected in the SpanSelector)

        # -- Find outliers from collected data
        x_sel_clean, y_sel_clean = reject_outliers(x_sel, y_sel)
        x_sel_outliers, y_sel_outliers = reject_outliers(x_sel, y_sel, switch_polarity=True)

        # -- Update plots on selection:
        self.update_plots(x_sel, y_sel, x_sel_clean, y_sel_clean, x_sel_outliers, y_sel_outliers)
        self.fig.canvas.draw()

    def update_plots(self, x_sel, y_sel, x_sel_clean, y_sel_clean, x_sel_outliers, y_sel_outliers):

        self.ax1.cla()
        self.ax1.plot(self.fullx, self.fully)  # -- Replot full data because we have called cla()
        self. ax1.set_xticklabels(convert_labView_timestamp(self.ax1.get_xticks()), rotation=0)
        # -- Update selected regions visualization
        for ind_range in self.intervals:
            self.ax1.axvspan(self.fullx[ind_range[0]], self.fullx[ind_range[1]], alpha=0.3, facecolor='red')

        self.line2.set_data(x_sel_clean, y_sel_clean)  # update clean data
        self.line2a.set_data(x_sel_outliers, y_sel_outliers)  # update outliers data
        self.ax2.set_xlim(x_sel[0], x_sel[-1])
        self.ax2.set_ylim(y_sel.min(), y_sel.max())
        self.ax2.set_xticklabels(convert_labView_timestamp(self.ax2.get_xticks()), rotation=0)

        self.ax3.cla()
        sns.distplot(y_sel_clean, kde=True, norm_hist=True, hist_kws={'alpha': 0.75, 'rwidth': 0.9}, color='#f37736',
                     ax=self.ax3)

        taus = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        data_sel_clean = np.vstack((x_sel_clean, y_sel_clean)).T

        t2, this_ad_ff, this_err_lo_ff, this_err_hi_ff, _ = oadev(data_sel_clean, taus, alpha=0)

        adev_legend = print_summary(t2, this_ad_ff, this_err_lo_ff, this_err_hi_ff)

        # calculate clock stability from straight-line fit to adev white noise
        x_fit, y_fit, a, b = fit_adev(t2, this_ad_ff, this_err_lo_ff, this_err_hi_ff)

        # sigma = 10^b * tau^a; assuming white noise i.e. a = 0.5
        instability_per_sqrt_tau = 10**b
        print('\nOADEV / sqrt(tau) @ 1s:', instability_per_sqrt_tau)

        self.ax4.cla()
        self.ax4.errorbar(x=t2, y=this_ad_ff, yerr=[this_err_lo_ff, this_err_hi_ff], ls='--', marker='o', capsize=2,
                          elinewidth=1,
                          markeredgewidth=1, c='C3')

        self.ax4.plot(x_fit, y_fit, ls='--', c='C4')  # plot fit to adev tail
        self.ax4.text(0.8, 0.99, adev_legend, horizontalalignment='left', verticalalignment='top', transform=self.ax4.transAxes)  # add legend as textbox

        instability_text = f"Instability: {instability_per_sqrt_tau * 10 ** 16:.2f} e-16 Hz/sqrt(tau)"
        self.ax4.text(0.075, 0.9, instability_text, horizontalalignment='left', verticalalignment='top', transform=self.ax4.transAxes)

        self.ax4.set_yscale('log')
        self.ax4.set_xscale('log')

        self.ax4.set_xlim(1, 10 ** 5)
        self.ax4.set_ylim(10 ** (-18), 10 ** (-15))

        # self.ax4.set_title('Frequency Stability')
        self.ax4.set_xlabel(r'Averaging Time, $\tau$, Seconds')
        self.ax4.set_ylabel(r'Overlapping Allan Deviation, $\sigma_y(\tau)$')

        self.ax4.grid(which='major', ls='--')
        self.ax4.grid(which='minor', ls=':', c='LightGray')

        # or could go smarter way and just update line and not whole axis
        # self.line4.set_data(t2, this_ad_ff)
        # self.caplines4.set_data(this_err_lo_ff, this_err_hi_ff)
        # Add [this_err_lo_ff, this_err_hi_ff]


        # -- Return statistics on selection
        ff_shift = np.mean(y_sel_clean) * 1 / 429228004229873  # clock shift
        # Error on fractional frequency offset: cannot use a naive standard error of the mean as samples not iid (e.g.
        # at short taus). Instead extrapolate white frequency noise @1s and evaluate it at sqrt(max tau). Max tau can
        # be last tau at which we computed adev to be conservative about 'unknown' noise behaviour at unseen taus, or
        # max tau can be last time in dataset if know white frequency noise behaviour is valid up until there. Also see
        # Benkler, E., Lisdat, C. and Sterr, U., 2015 showing that for white frequency noise ff_shift var = adev**2.
        ff_shift_std = 10**b/np.sqrt(t2[-1])  # conservative estimate

        self.ax3.axvline(x=ff_shift, ls='--', c='Gray')
        ff_shift_text = f"\nFractional frequency offset:\n{ff_shift*10**19:.2f} +- {ff_shift_std*10**19:.2f} e-19 Hz"
        self.ax3.text(0.05, 0.999, ff_shift_text, horizontalalignment='left', verticalalignment='top', transform=self.ax3.transAxes)


def launch_gui(data):
    """Launches an interactive window to human-select good data chunks. Then algorithm removes extra outliers in chunk.

    Args:
        data (np.array):    array of x,y data (n,2)

    Returns:
        clean_data (np.array): selected array of x,y data (n,2) with outliers removed
    """
    # Extract data columns from dataframe
    x = data[:, 0]
    y = data[:, 1]

    # Blueprint, setup subplots layout
    fig = plt.figure(figsize=(15, 15))  # window aspect ration. Make sure matches aspect ratio of GridSpec below
    gs1 = GridSpec(4, 4)  # num_rows, num_columns
    gs1.update(wspace=0.5, hspace=0.35)

    ax1 = fig.add_subplot(gs1[0, :])    # axes for full data
    ax2 = fig.add_subplot(gs1[1, :])    # axes for selected data
    ax3 = fig.add_subplot(gs1[2:, :2])  # axes for histogram
    ax4 = fig.add_subplot(gs1[2:, 2:])  # axes for Allan plot

    # Plot full data
    ax1.plot(x, y)
    ax1.set_xticklabels(convert_labView_timestamp(ax1.get_xticks()), rotation=0)


    # Initialize placeholder for discarded/selected data
    line2, = ax2.plot([], [])  # clean selected data
    line2a, = ax2.plot([], [], c='r', linestyle='None', marker="x")  # discarded selected data

    # Initialize histogram
    sns.distplot([], kde=True, norm_hist=True, hist_kws={'alpha': 0.75, 'rwidth': 0.9}, color='#f37736', ax=ax3)

    # Initialize Allan Plot
    line4, _, _ = ax4.errorbar(x=[], y=[], yerr=[], ls='--', marker='o', capsize=2, elinewidth=1, markeredgewidth=1, c='C3')
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    #ax4.set_title('Frequency Stability')
    ax4.set_xlabel(r'Averaging Time, $\tau$, Seconds')
    ax4.set_ylabel(r'Overlapping Allan Deviation, $\sigma_y(\tau)$')
    ax4.grid(which='major', ls='--')
    ax4.grid(which='minor', ls=':', c='LightGray')

    # Launch selector widget
    onselect = Onselect(x, y, fig, ax1, ax2, line2, line2a, ax3, line4, ax4)  # Create onselect object which will collect the selected data

    span = SpanSelector(ax1, onselect, 'horizontal', useblit=True, rectprops=dict(alpha=0.3, facecolor='red'))

    plt.show()  # wait for user to close plot

    return 1


if __name__ == '__main__':

    # Path to a .txt file containing timestamps as first column, and frequency data as second column
    # Here we assume timestamps are relative to LabView epoch, but can change
    filename = "path/to/data.txt"

    # Skip header
    data = np.loadtxt(filename, skiprows=0)

    # Can concatenate multiple data files with np.concatenate:
    # data = np.concatenate((data1, data2, ...))
    
    # GUI Start
    launch_gui(data)