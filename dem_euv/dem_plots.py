import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import corner
from scipy.interpolate import interp1d
from typing import List, Callable, Any, Union, Tuple
from numpy.polynomial.chebyshev import chebval
from scipy.integrate import cumtrapz
from astropy.io import fits

__all__ = ['plot_dem', 'plot_spectrum', 'display_fig']

def plot_dem(samples, lnprob, flux_arr, gofnt_matrix,
             log_temp, temp, flux_weighting,
             main_color, sample_color, alpha,
             sample_num, sample_label,
             main_label, title_name, figure=None,
             low_y=19.0, high_y=31.0):
    if figure is not None:
        plt.figure(figure.number)

    temp_interp = interp1d(np.arange(0,len(temp),1), temp)
    temp = temp_interp(np.linspace(0,len(temp)-1,len(gofnt_matrix[0])))

    shift_log_temp = log_temp - np.mean(log_temp)
    range_temp = (np.max(shift_log_temp) - np.min(shift_log_temp))
    shift_log_temp /= (0.5 * range_temp)
    psi_model = 10.0**chebval(shift_log_temp, samples[np.argmax(lnprob)])
    total_samples = np.random.choice(len(samples), sample_num)

    psi_ys = flux_arr / (flux_weighting * np.trapz(gofnt_matrix, temp))

    temp_lows = np.min(temp) * np.ones((len(gofnt_matrix)))
    temp_upps = np.max(temp) * np.ones_like(temp_lows)
    temp_lows = 1e4 * np.ones_like(flux_arr)#psi_ys)
    temp_upps = 1e8 * np.ones_like(temp_lows)
    for i in range(len(flux_arr)):
        gofnt_cumtrapz = cumtrapz(gofnt_matrix[i], temp)
        low_index = np.argmin(
            np.abs(gofnt_cumtrapz - (0.16 * gofnt_cumtrapz[-1])))
        upp_index = np.argmin(
            np.abs(gofnt_cumtrapz - (0.84 * gofnt_cumtrapz[-1])))
        temp_lows[i] = temp[low_index + 1]
        temp_upps[i] = temp[upp_index + 1]

    psi_samples = np.zeros(sample_num, dtype=np.ndarray)
    for i in range(0, sample_num):
        s = samples[total_samples[i]]
        temp_psi = 10.0**chebval(shift_log_temp, s)
        
        psi_interp = interp1d(np.arange(0,len(temp_psi),1), temp_psi)
        temp_psi   = psi_interp(np.linspace(0, len(temp_psi)-1, len(temp)))

        if i == 0:
            plt.loglog(temp, temp_psi,
                       color=sample_color, alpha=alpha, label=sample_label)
        else:
            plt.loglog(temp, temp_psi, color=sample_color, alpha=alpha)
        psi_samples[i] = temp_psi

    m_interp = interp1d(np.arange(0,len(psi_model),1), psi_model)

    plt.loglog(temp, m_interp(np.linspace(0,len(psi_model)-1,len(temp))),
               color=main_color, label=main_label)

    plt.hlines(psi_ys, temp_lows, temp_upps, label='Flux Constraints',
               colors='k', zorder=100)

    plt.ylim(10.0**low_y, 10.0**high_y)
    plt.xlabel('Temperature [K]')
    y_label = r'$\Psi(T) = N_e N_{\mathrm{H}} \frac{ds}{dT}$ '
    y_label += r'[cm$^{-5}$ K$^{-1}$]'
    plt.ylabel(y_label)
    plt.title(title_name)
    return plt.gcf(), [temp, m_interp(np.linspace(0,len(psi_model)-1,len(temp))),
                       psi_ys, temp_lows, temp_upps, psi_samples]


def plot_spectrum(spec_fits, title_name,
                  figure=None, alpha=0.3, color='b'):
    hdu = fits.open(spec_fits)
    wave = hdu[1].data['Wavelength']
    flux = hdu[1].data['Flux_density']
    upp = flux + hdu[1].data['Upper_Error_84']
    low = flux - hdu[1].data['Lower_Error_16']
    plt.semilogy(wave, flux, drawstyle='steps-mid', color=color)
    plt.fill_between(wave, low, upp, color=color, alpha=alpha)
    plt.title(title_name)
    plt.xlabel(r'Wavelength [$\mathrm{\AA}$]')
    plt.ylabel(r'Flux [erg s$^{-1}$ cm$^{-2}$ $\mathrm{\AA}^{-1}$]')
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
