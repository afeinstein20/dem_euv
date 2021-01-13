import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import corner
from typing import List, Callable, Any, Union, Tuple
from numpy.polynomial.chebyshev import chebval


def plot_dem(samples, lnprob, flux_arr, gofnt_matrix,
             log_temp, temp, flux_weighting,
             main_color, sample_color, alpha,
             sample_num, sample_label,
             main_label, title_name, figure=None,
             low_y=19.0, high_y=31.0):
    if figure is not None:
        plt.figure(figure.number)
    shift_log_temp = log_temp - np.mean(log_temp)
    range_temp = (np.max(shift_log_temp) - np.min(shift_log_temp))
    shift_log_temp /= (0.5 * range_temp)
    psi_model = 10.0**chebval(shift_log_temp, samples[np.argmax(lnprob)])
    total_samples = np.random.choice(len(samples), sample_num)
    psi_ys = flux_arr / (flux_weighting * np.trapz(gofnt_matrix, temp))
    temp_integrands = [(psi_model * gofnt)
                       for gofnt in gofnt_matrix]
    temp_lows = np.min(temp) * np.ones((len(gofnt_matrix)))
    temp_upps = np.max(temp) * np.ones_like(temp_lows)
    for i in range(len(gofnt_matrix)):
        temp_integrand = temp_integrands[i]
        temp_halfmax = 0.5 * np.max(temp_integrand)
        if np.argmax(temp_integrand) > 0:
            temp_diff_low = np.abs(temp_integrand[:np.argmax(temp_integrand)]
                                   - temp_halfmax)
            temp_diff_low_arg = np.argmin(temp_diff_low)
            temp_lows[i] = temp[:np.argmax(temp_integrand)][temp_diff_low_arg]
        if np.argmax(temp_integrand) < (len(temp) - 1):
            temp_diff_upp = np.abs(temp_integrand[np.argmax(temp_integrand):]
                                   - temp_halfmax)
            temp_diff_upp_arg = np.argmin(temp_diff_upp)
            temp_upps[i] = temp[np.argmax(temp_integrand):][temp_diff_upp_arg]
    for i in range(0, sample_num):
        s = samples[total_samples[i]]
        temp_psi = 10.0**chebval(shift_log_temp, s)
        if i == 0:
            plt.loglog(temp, temp_psi,
                       color=sample_color, alpha=alpha, label=sample_label)
        else:
            plt.loglog(temp, temp_psi, color=sample_color, alpha=alpha)
    plt.loglog(temp, psi_model, color=main_color, label=main_label)
    plt.hlines(psi_ys, temp_lows, temp_upps, label='Flux Constraints',
               colors='k', zorder=100)
    plt.ylim(10.0**low_y, 10.0**high_y)
    plt.xlabel('Temperature [K]')
    y_label = r'$\Psi(T) = N_e N_{\mathrm{H}} \frac{ds}{dT}$ '
    y_label += r'[cm$^{-5}$ K$^{-1}$]'
    plt.ylabel(y_label)
    plt.title(title_name)
    return plt.gcf()


def display_fig(figure, name, mode='pdf', dpi=500, legend=True):
    plt.figure(figure.number)
    if legend is True:
        plt.legend()
    plt.tight_layout()
    if mode == 'SHOW':
        plt.show()
    else:
        plt.savefig(name + '.' + mode, dpi=dpi)
        plt.clf()
    return figure
