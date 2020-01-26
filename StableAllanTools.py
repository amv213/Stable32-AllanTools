import allantools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_better_ade(tau, adev, tau0, N, alpha=0, d=2, overlapping=True, modified=False):
    """
    Calculate non-naive Allan deviation errors. Equivalent to Stable32
    https://github.com/aewallin/allantools/blob/master/examples/ci_demo.py
    https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20050061319.pdf

    tau0      averaging factor;  interval between measurements
    N         number of frequency observations
    alpha     +2,...,-4   noise type, either estimated or known
    d         1 first-difference variance, 2 allan variance, 3 hadamard variance; we require: alpha+2*d >1
    """

    # Confidence-intervals for each (tau, adev) pair separately.
    cis = []
    for (t, dev) in zip(tau, adev):
        # Greenhalls EDF (Equivalent Degrees of Freedom)
        edf = allantools.edf_greenhall(alpha=alpha, d=d, m=t / tau0, N=N, overlapping=overlapping, modified=modified)
        # with the known EDF we get CIs
        (lo, hi) = allantools.confidence_interval(dev=dev, edf=edf)
        cis.append((lo, hi))

    err_lo = np.array([d - ci[0] for (d, ci) in zip(ad, cis)])
    err_hi = np.array([ci[1] - d for (d, ci) in zip(ad, cis)])

    # now we are ready to print and plot the results
    print("Tau\tmin Dev\t\tDev\t\tMax Dev")
    for (tau, dev, ci) in zip(tau, adev, cis):
        print("%d\t%f\t%f\t%f" % (tau, ci[0], dev, ci[1]))

    return err_lo, err_hi


if __name__ == '__main__':

    plt.rc('font', size=12)

    data = np.loadtxt('MYDATA.txt', skiprows=0)
    x = data[:, 0]  # Get timestamps
    y = data[:, 1]  # Get fractional frequency data

    #sns.distplot(y, rug=True, kde=True, norm_hist=True, hist_kws={'alpha': 0.75, 'rwidth': 0.9}, color='#f37736')
    #plt.show()

    avg_interval = (x[-1]-x[0])/len(x)  # average interval between measurements
    r = 1/avg_interval  # average sample rate in Hz of the input data

    t = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])*avg_interval  # tau values on which to evaluate metric
    (t2, ad, ade, adn) = allantools.oadev(y, rate=r, data_type="freq", taus=t)  # normal ODEV computation, giving naive 1/sqrt(N) errors

    # correct for deadtime: ad/np.sqrt(B2*B3)
    # https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication1065.pdf | 5.15 Dead Time
    # TODO

    # Correct (Stable32) errors
    err_lo, err_hi = get_better_ade(t2, ad, avg_interval, len(x))

    # Rescale
    scale = 1/429228004229873  # for fractional frequency instability; e.g. 1/Sr87_BIPM_frequency
    ad_ff = ad*scale
    err_lo_ff = err_lo*scale
    err_hi_ff = err_hi*scale

    # Plot
    plt.figure()
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')

    plt.errorbar(x=t2, y=ad_ff, yerr=[err_lo_ff, err_hi_ff], ls='--', marker='o', capsize=2, elinewidth=1, markeredgewidth=1, c='C3')  # Plot the results
    plt.plot(x=t2, y=t2*10**(-16))

    plt.xlim(1, 10**3)
    plt.ylim(10**(-17), 10**(-15))

    plt.title('Frequency Stability')
    plt.xlabel(r'Averaging Time, $\tau$, Seconds')
    plt.ylabel(r'Overlapping Allan Deviation, $\sigma_y(\tau)$')

    plt.grid(which='major', ls='--')
    plt.grid(which='minor', ls=':', c='LightGray')

    plt.show()
