import corner
import emcee
import os.path
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from data_prep import generate_constant_R_wave_arr
from data_prep import generate_spectrum_from_samples
from gofnt_routines import parse_ascii_table_CHIANTI, resample_gofnt_matrix
from gofnt_routines import generate_ion_gofnts, do_gofnt_matrix_integral
from astropy.table import Table
from dem_plots import plot_dem, plot_spectrum, display_fig
from scipy.io import readsav
from resample import bintogrid
from multiprocessing import Pool
from numpy.polynomial.chebyshev import chebval
from multiprocessing import cpu_count
from scipy.integrate import cumtrapz


star_name_root = 'sun_euv'
abundance = 0.0
abund_type = 'sol0'
star_rad = 1.0 * u.Rsun
star_dist = 1.0 * u.AU

line_table_file = 'sun_euv_linetable.ascii'
data_npy_name = 'sun_spectrum_data.npy'

press_list = [1e14, 1e15, 1e16, 1e17, 1e18, 1e19][::-1]

burn_in_steps = 1000
double_burn = True
production_steps = 3000
count_print = True
n_walkers = 50
init_spread = 1e-3
second_spread = 1e-4
buffer = 4
sample_num = int(1e5)

def generate_data_npy():
    sun_sav = readsav('Solar-data.idlsav')
    wave = 10 * np.array(sun_sav['wave']).flatten()
    flux = 100 * np.array(sun_sav['flux']).flatten()
    err = 0.01 * flux
    bins = np.diff(wave)
    bins = np.append(bins[0], bins)

    mask = np.where((wave >= 1.0) & (wave <= 100.0) & np.isfinite(flux))

    xray_flux = flux[mask]
    xray_err = err[mask]
    xray_wave = wave[mask]
    xray_bins = bins[mask]

    np.save(data_npy_name,
            [xray_wave, xray_bins, xray_flux, xray_err])
    return data_npy_name


def generate_flux_weighting(star_name, star_dist, star_rad):
    flux_weighting = ((np.pi * u.sr * (star_rad**2.0) *
                       (1.0 / (star_dist**2.0))).to(u.sr)).value
    np.save('flux_weighting_' + star_name, [flux_weighting])
    return flux_weighting


def get_best_gofnt_matrix_press(abundance, press, abund_type):
    home_dir = os.path.expanduser('~')
    gofnt_dir = home_dir + '/repos_all/research/gofnt_dir/'
    gofnt_root = 'gofnt_w1_w1500_t4_t8_r100_p'
    gofnt_matrix = np.load(gofnt_dir + gofnt_root
                           + str(int(np.log10(press)))
                           + '_' + abund_type + '.npy')
    if abund_type == 'sol0':
        if abundance != 0.0:
            gofnt_matrix *= 10.0**abundance
    return gofnt_matrix


def get_line_data_gofnts(star_name, line_table_file, abundance,
                         temp, dens, bin_width):
    line_table = parse_ascii_table_CHIANTI(line_table_file)
    gofnt_lines, flux, err, names = generate_ion_gofnts(line_table,
                                                        abundance,
                                                        bin_width,
                                                        temp,
                                                        dens)
    np.save('gofnt_lines_' + star_name + '.npy', gofnt_lines)
    np.save('ion_fluxes_' + star_name + '.npy', flux)
    np.save('ion_errs_' + star_name + '.npy', err)
    np.save('ion_names_' + star_name + '.npy', names)
    return gofnt_lines, flux, err, names


def get_spectrum_data_gofnt(star_name, data_npy_file, gofnt_matrix):
    wave, wave_bins, flux, err = np.load(data_npy_file)
    flux *= wave_bins
    err *= wave_bins
    wave_old, _ = generate_constant_R_wave_arr(1, 1500, 100)
    gofnt_spectrum = resample_gofnt_matrix(gofnt_matrix, wave, wave_bins,
                                           wave_old)
    np.save('gofnt_spectrum_' + star_name + '.npy', gofnt_spectrum)
    np.save('spectrum_fluxes_' + star_name + '.npy', flux)
    np.save('spectrum_errs_' + star_name + '.npy', err)
    np.save('spectrum_wave_' + star_name + '.npy', wave)
    temp = np.logspace(4, 8, 2000)
    temp_lows = 1e4 * np.ones_like(flux)
    temp_upps = 1e8 * np.ones_like(temp_lows)
    for i in range(len(flux)):
        gofnt_cumtrapz = cumtrapz(gofnt_spectrum[i], temp)
        low_index = np.argmin(
            np.abs(gofnt_cumtrapz - (0.25 * gofnt_cumtrapz[-1])))
        upp_index = np.argmin(
            np.abs(gofnt_cumtrapz - (0.75 * gofnt_cumtrapz[-1])))
        temp_lows[i] = temp[low_index + 1]
        temp_upps[i] = temp[upp_index + 1]
    gofnt_filter = np.where((temp_upps <= 3e7))
    print(gofnt_filter)
    # for i in range(len(gofnt_filter[0])):
    #    plt.loglog(temp, gofnt_spectrum[gofnt_filter[0][i]])
    # plt.ylim(1e-27, 1e-24)
    # plt.show()
    return gofnt_spectrum[gofnt_filter], flux[gofnt_filter], err[gofnt_filter]


def get_star_data_gofnt_press(star_name, abundance, press,
                              line_table_file=None, data_npy_file=None,
                              bin_width=1.0):
    big_gofnt = get_best_gofnt_matrix_press(abundance, press, 'sol0')
    temp = np.logspace(4, 8, 2000)
    dens = press / temp
    if line_table_file is not None:
        if os.path.isfile('gofnt_lines_' + star_name + '.npy'):
            gofnt_lines = np.load('gofnt_lines_' + star_name + '.npy')
            flux = np.load('ion_fluxes_' + star_name + '.npy')
            err = np.load('ion_errs_' + star_name + '.npy')
        else:
            gofnt_lines, flux, err, _ = get_line_data_gofnts(star_name,
                                                             line_table_file,
                                                             abundance,
                                                             temp, dens,
                                                             bin_width)
        line_flux = flux
        line_err = err
    else:
        gofnt_lines = None
    if data_npy_file is not None:
        if os.path.isfile('gofnt_spectrum_' + star_name + '.npy'):
            gofnt_spectrum = np.load('gofnt_spectrum_' + star_name + '.npy')
            flux = np.load('spectrum_fluxes_' + star_name + '.npy')
            err = np.load('spectrum_errs_' + star_name + '.npy')
        else:
            gofnt_spectrum, flux, err = get_spectrum_data_gofnt(star_name,
                                                                data_npy_file,
                                                                big_gofnt)
        spectrum_flux = flux
        spectrum_err = err
    else:
        gofnt_spectrum = None
    if (gofnt_lines is None):
        if (gofnt_spectrum is None):
            print('Where is this star\'s data to do anything with?')
        else:
            gofnt_matrix = gofnt_spectrum
            np.save('gofnt_' + star_name + '.npy', gofnt_matrix)
            np.save('flux_' + star_name + '.npy', flux)
            np.save('err_' + star_name + '.npy', err)
    elif (gofnt_spectrum is None):
        gofnt_matrix = gofnt_lines
        np.save('gofnt_' + star_name + '.npy', gofnt_matrix)
        np.save('flux_' + star_name + '.npy', flux)
        np.save('err_' + star_name + '.npy', err)
    else:
        gofnt_matrix = np.append(gofnt_spectrum, gofnt_lines, axis=0)
        flux = np.append(spectrum_flux, line_flux)
        err = np.append(spectrum_err, line_err)
        np.save('gofnt_' + star_name + '.npy', gofnt_matrix)
        np.save('flux_' + star_name + '.npy', flux)
        np.save('err_' + star_name + '.npy', err)
    return gofnt_matrix, flux, err


def ln_prior_func(params):
    flux_factor = params[-1]
    coeffs = params[:-1]
    lp = 0.0
    if coeffs[0] >= 29:
        lp += 29 - coeffs[0]
    elif coeffs[0] <= 17:
        lp += coeffs[0] - 17
    if chebval(0.0, coeffs) <= 0.0:
        return -np.inf
    else:
        pass
    for coeff in coeffs:
        if coeff >= -10.0**2.0:
            if coeff <= 10.0**2.0:
                pass
            else:
                return -np.inf
        else:
            return -np.inf
    if flux_factor < -2.0:
        return -np.inf
    elif flux_factor > 2.0:
        return -np.inf
    elif chebval(-1.0, coeffs) <= chebval(-0.99, coeffs):
        return -np.inf
    else:
        return lp


def ln_prob_func(params):
    flux_factor = 10.0**(params[-1])
    coeffs = params[:-1]
    shift_log_temp = log_temp - np.mean(log_temp)
    range_temp = (np.max(shift_log_temp) - np.min(shift_log_temp))
    shift_log_temp /= (0.5 * range_temp)
    cheb_polysum = chebval(shift_log_temp, coeffs)

    psi_model = 10.0**cheb_polysum
    if np.nanmin(psi_model) <= 0:
        return -np.inf
    model = do_gofnt_matrix_integral(psi_model, gofnt_matrix,
                                     temp, flux_weighting)
    var_term = (((flux_factor * model)**2) + (yerr ** 2))
    lead_term = np.log(1.0 / np.sqrt(2.0 * np.pi * var_term))
    inv_var = 1.0 / var_term
    val = np.sum(lead_term - (0.5 * ((((y - model)**2) * inv_var))))
    if np.isfinite(val):
        return val
    return -np.inf


def likelihood_func(params):
    lp = ln_prior_func(params)
    if np.isfinite(lp):
        return lp + ln_prob_func(params)
    return -np.inf


def fit_emcee(init_pos):
    ndim = len(init_pos)
    pos = [init_pos + init_spread * np.random.randn(ndim) * init_pos
           for i in range(n_walkers)]
    print("Initializing walkers")
    print(cpu_count())
    with Pool(cpu_count() - buffer) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, ndim, likelihood_func,
                                        pool=pool)
        print("Starting burn-in")
        p0, prob, _ = sampler.run_mcmc(pos, burn_in_steps,
                                       progress=count_print)
        p0 = [p0[np.argmax(prob)] + second_spread * np.random.randn(ndim) *
              p0[np.argmax(prob)] for i in range(n_walkers)]
        sampler.reset()
        nsteps = burn_in_steps + production_steps
        done_steps = burn_in_steps
        if double_burn:
            print('Starting second burn-in')
            nsteps += burn_in_steps
            done_steps += burn_in_steps
            p0, prob, _ = sampler.run_mcmc(p0, burn_in_steps,
                                           progress=count_print)
            p0 = [p0[np.argmax(prob)] + second_spread * np.random.randn(ndim) *
                  p0[np.argmax(prob)] for i in range(n_walkers)]
            sampler.reset()
        print('Starting production')
        sampler.run_mcmc(p0, production_steps,
                         progress=count_print)
    return sampler.flatchain, sampler.flatlnprobability, sampler


temp = np.logspace(4, 8, 2000)
log_temp = np.log10(temp)
shift_log_temp = log_temp - np.mean(log_temp)
range_temp = (np.max(shift_log_temp) - np.min(shift_log_temp))
shift_log_temp /= (0.5 * range_temp)

flux_weighting = generate_flux_weighting(star_name_root,
                                         star_dist, star_rad)
init_pos = [22.49331207, -3.31678227, -0.49848262,
            -1.27244452, -0.93897032, -0.67235648,
            -0.08085897]

if os.path.isfile(data_npy_name):
    data_npy_file = data_npy_name
else:
    data_npy_file = generate_data_npy()

for press in press_list:
    star_name = star_name_root + '_p' + str(int(np.log10(press)))
    gofnt_all = get_best_gofnt_matrix_press(abundance, press, abund_type)
    wave_all, bin_all = generate_constant_R_wave_arr(1, 1500, 100)
    if os.path.isfile('gofnt_' + star_name + '.npy'):
        gofnt_matrix = np.load('gofnt_' + star_name + '.npy')
        flux = np.load('flux_' + star_name + '.npy')
        err = np.load('err_' + star_name + '.npy')
    else:
        out = get_star_data_gofnt_press(star_name,
                                        abundance,
                                        press,
                                        line_table_file,
                                        data_npy_file)
        gofnt_matrix, flux, err = out
    if os.path.isfile('samples_' + star_name + '.npy'):
        samples = np.load('samples_' + star_name + '.npy')
        lnprob = np.load('lnprob_' + star_name + '.npy')
    else:
        y = flux
        yerr = err
        samples, lnprob, _ = fit_emcee(init_pos)
        np.save('samples_' + star_name, samples)
        np.save('lnprob_' + star_name, lnprob)
    if os.path.isfile('dem_' + star_name + '.pdf'):
        pass
    else:
        g = plot_dem(samples[:, :-1], lnprob, flux, gofnt_matrix,
                     log_temp, temp, flux_weighting,
                     'b', 'cornflowerblue', 0.1, 500,
                     'MCMC Samples', 'Best-fit DEM model',
                     r'$\Psi(T)$')
        g = display_fig(g, 'dem_' + star_name, mode='pdf')
        plt.clf()

    if os.path.isfile('corner_' + star_name + '.pdf'):
        pass
    else:
        h = corner.corner(samples, quantiles=[0.16, 0.5, 0.84],
                          show_titles=True, title_kwargs={"fontsize": 12},
                          plot_contours=True)
        h = display_fig(h, 'corner_' + star_name, mode='pdf')
        plt.clf()

    if os.path.isfile('spectrum_' + star_name + '.fits'):
        spectrum_table = Table.read('spectrum_' + star_name + '.fits')

    else:
        spectrum_name = 'spectrum_' + star_name
        best_psi = 10.0**chebval(shift_log_temp,
                                 samples[np.argmax(lnprob)][:-1])

        best_spectra2 = do_gofnt_matrix_integral(best_psi, gofnt_all,
                                                 temp,
                                                 flux_weighting)
        best_spectra2 /= bin_all

        all_indices = np.arange(0, len(samples))
        rand_indices = np.random.choice(all_indices, sample_num)
        spec_len = len(wave_all)
        all_spectra = np.zeros((sample_num, spec_len))
        all_psi = np.zeros((sample_num, len(temp)))

        def fill_psi(i):
            temp_psi = 10.0**(chebval(shift_log_temp,
                                      samples[rand_indices[i], :][:-1]))
            temp_spectra = do_gofnt_matrix_integral(temp_psi, gofnt_all,
                                                    temp, flux_weighting)
            temp_err = temp_spectra * (10.0**(samples[rand_indices[i]][-1]))
            rand_spectra = np.random.normal(loc=temp_spectra, scale=temp_err)
            rand_spectra /= bin_all
            return rand_spectra, temp_psi

        with Pool(cpu_count() - buffer) as pool:
            out = pool.map(fill_psi, range(sample_num))
        for i in range(sample_num):
            all_spectra[i, :] = out[i][0]
            all_psi[i, :] = out[i][1]

        wave_unit = u.Angstrom
        flux_unit = u.erg / (u.s * u.cm**2)
        best_spectra = np.median(all_spectra, axis=0)

        med_psi = np.median(all_psi, axis=0)
        upp_psi = np.percentile(all_psi, 84, axis=0)
        low_psi = np.percentile(all_psi, 16, axis=0)

        upper_diff_var = (np.percentile(
            all_spectra, 84, axis=0) - best_spectra)**2
        lower_diff_var = (
            best_spectra - np.percentile(all_spectra, 16, axis=0))**2

        upper_var = upper_diff_var
        lower_var = lower_diff_var
        upper_err = np.sqrt(upper_var)
        lower_err = np.sqrt(lower_var)

        spectrum_table = Table([wave_all * wave_unit,
                                best_spectra * flux_unit,
                                lower_err * flux_unit,
                                upper_err * flux_unit,
                                best_spectra2 * flux_unit],
                               names=('Wavelength', 'Flux_density',
                                      'Lower_Error_16', 'Upper_Error_84',
                                      'Flux_density_ln_lmax'))
        spectrum_table.write(spectrum_name + '.fits',
                             format='fits', overwrite=True)
        np.save(spectrum_name + '_dems.npy', [low_psi, med_psi, upp_psi])
    if os.path.isfile('spectrum_' + star_name + '.pdf'):
        pass
    else:
        spec_fig = plot_spectrum('spectrum_' + star_name + '.fits',
                                 'DEM-generated Spectrum',)
        spec_fig = display_fig(spec_fig, 'spectrum_' + star_name,
                               mode='pdf')
