import corner
import os.path
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.table import Table

from resample import bintoR
from data_prep import generate_constant_bin_wave_arr, generate_constant_R_wave_arr
from data_prep import generate_spectrum_from_samples
from gofnt_routines import parse_ascii_table_CHIANTI, resample_gofnt_matrix
from gofnt_routines import generate_ion_gofnts
from fitting import fit_emcee, ln_likelihood_dem
from dem_plots import plot_dem, plot_spectrum, display_fig

OUTPUT_DIR = './flare/'

__all__ = ['generate_flux_weighting', 'get_best_gofnt_matrix_press',
           'get_line_data_gofnts', 'get_spectrum_data_gofnt',
           'get_star_data_gofnt_press', 'run_mcmc_single_star',
           'generate_spectrum_data_npy']


def generate_flux_weighting(star_name, star_dist, star_rad, suffix='_q'):
    flux_weighting = ((np.pi * u.sr * (star_rad**2.0) *
                       (1.0 / (star_dist**2.0))).to(u.sr)).value
    np.save('flux_weighting_' + star_name + suffix, [flux_weighting])
    return flux_weighting


def get_best_gofnt_matrix_press(abundance, press, abund_type=15,
                                gofnt_dir = '../gofnt_dir/'):
    gofnt_root = 'gofnt_w1_w2000_t4_t8_n100_p'
#    print(str(int(np.log10(press))))
#    gofnt_matrix = np.load(gofnt_dir + gofnt_root
#                           + str(int(np.log10(press)))
#                           + '_' + str(abund_type) + '.npy')

    gofnt_matrix = np.load(os.path.join(gofnt_dir, 'gofnt_w1_w1500_t4_t8_n100_r100_p17_sol0.npy'))
    gofnt_matrix *= 10.0**abundance
    return gofnt_matrix


def get_line_data_gofnts(star_name, line_table_file, abundance,
                         temp, dens, bin_width, suffix):
    global OUTPUT_DIR
    line_table = parse_ascii_table_CHIANTI(line_table_file)
    gofnt_lines, flux, err, names = generate_ion_gofnts(line_table,
                                                        abundance,
                                                        bin_width,
                                                        temp,
                                                        dens)
    np.save(OUTPUT_DIR+'gofnt_lines_' + star_name + suffix + '.npy', gofnt_lines)
    np.save(OUTPUT_DIR+'ion_fluxes_' + star_name + suffix + '.npy', flux)
    np.save(OUTPUT_DIR+'ion_errs_' + star_name + suffix + '.npy', err)
    np.save(OUTPUT_DIR+'ion_names_' + star_name + suffix + '.npy', names)
    return gofnt_lines, flux, err, names


def get_spectrum_data_gofnt(star_name, data_npy_file, gofnt_matrix, suffix):
    global OUTPUT_DIR

    wave, wave_bins, flux, err = np.load(data_npy_file, allow_pickle=True)
    flux *= wave_bins
    err *= wave_bins
    wave_old, _ = generate_constant_R_wave_arr(1, 1500, 150)

    gofnt_spectrum = resample_gofnt_matrix(gofnt_matrix, wave, wave_bins,
                                           wave_old)
    temp = np.logspace(4, 8, 100)
    gofnt_ints = np.trapz(gofnt_spectrum, temp)
    mask = np.where(gofnt_ints >=
                    (np.max(gofnt_ints) / 10))
    print(len(mask[0]))
    np.save(OUTPUT_DIR+'gofnt_spectrum_' + star_name + suffix+ '.npy', gofnt_spectrum[mask[0], :])
    np.save(OUTPUT_DIR+'spectrum_fluxes_' + star_name + suffix+ '.npy', flux[mask])
    np.save(OUTPUT_DIR+'spectrum_errs_' + star_name +suffix+ '.npy', err[mask])
    np.save(OUTPUT_DIR+'spectrum_waves_' + star_name +suffix+ '.npy', wave[mask])
    np.save(OUTPUT_DIR+'spectrum_bins_' + star_name +suffix+'.npy', wave_bins[mask])
    return gofnt_spectrum, flux, err


def get_star_data_gofnt_press(star_name, abundance, press, 
                              line_table_file=None, data_npy_file=None,suffix='_q',
                              bin_width=1.0, abund_type=15, gofnt_dir='../gofnt_dir/'):
    global OUTPUT_DIR
    print('get_star_data: ', gofnt_dir)
    big_gofnt = get_best_gofnt_matrix_press(abundance, press, abund_type, gofnt_dir)
    temp = np.logspace(4, 8, 100)
    dens = press / temp

    if line_table_file is not None:
        if os.path.isfile(OUTPUT_DIR+'gofnt_lines_' + star_name+ suffix+'.npy'):
            gofnt_lines = np.load(OUTPUT_DIR+'gofnt_lines_' + star_name+suffix+ '.npy')
            flux = np.load(OUTPUT_DIR+'ion_fluxes_' + star_name+suffix+ '.npy')
            err = np.load(OUTPUT_DIR+'ion_errs_' + star_name +suffix+ '.npy')
        else:
            gofnt_lines, flux, err, _ = get_line_data_gofnts(star_name,
                                                             line_table_file,
                                                             abundance,
                                                             temp, dens,
                                                             bin_width, suffix=suffix)
        line_flux = flux
        line_err = err
    else:
        gofnt_lines = None
    if data_npy_file is not None:
        if os.path.isfile(OUTPUT_DIR+'gofnt_spectrum_' + star_name + suffix+'.npy'):
            gofnt_spectrum = np.load(OUTPUT_DIR+'gofnt_spectrum_' + star_name + suffix+'.npy')
            flux = np.load(OUTPUT_DIR+'spectrum_fluxes_' + star_name+suffix +'.npy')
            err = np.load(OUTPUT_DIR+'spectrum_errs_' + star_name+suffix +'.npy')
        else:
            gofnt_spectrum, flux, err = get_spectrum_data_gofnt(star_name,
                                                                data_npy_file,
                                                                big_gofnt, suffix=suffix)
        spectrum_flux = flux
        spectrum_err = err
    else:
        gofnt_spectrum = None

    if (gofnt_lines is None):
        if (gofnt_spectrum is None):
            print('Where is this star\'s data to do anything with?')
        else:
            gofnt_matrix = gofnt_spectrum
            np.save(OUTPUT_DIR+'gofnt_' + star_name + suffix +'.npy', gofnt_matrix)
            np.save(OUTPUT_DIR+'flux_' + star_name + suffix + '.npy', flux)
            np.save(OUTPUT_DIR+'err_' + star_name + suffix + '.npy', err)
    elif (gofnt_spectrum is None):
        gofnt_matrix = gofnt_lines
        np.save(OUTPUT_DIR+'gofnt_' + star_name +suffix + '.npy', gofnt_matrix)
        np.save(OUTPUT_DIR+'flux_' + star_name + suffix +'.npy', flux)
        np.save(OUTPUT_DIR+'err_' + star_name +  suffix +'.npy', err)
    else:
        gofnt_matrix = np.append(gofnt_spectrum, gofnt_lines, axis=0)
        flux = np.append(spectrum_flux, line_flux)
        err = np.append(spectrum_err, line_err)
        np.save(OUTPUT_DIR+'gofnt_' + star_name +suffix + '.npy', gofnt_matrix)
        np.save(OUTPUT_DIR+'flux_' + star_name + suffix + '.npy', flux)
        np.save(OUTPUT_DIR+'err_' + star_name +suffix + '.npy', err)
    return gofnt_matrix, flux, err


def run_mcmc_single_star(init_pos, gofnt_matrix, flux, err, flux_weighting,
                         n_walkers=200, burn_in_steps=1000,
                         production_steps=8000, thread_num=8,
                         count_num=2000, suffix='_q'):
    temp = np.logspace(4, 8, 100)
    log_temp = np.log10(temp)
    samples, lnprob, sampler = fit_emcee(init_pos=init_pos,
                                         likelihood_func=ln_likelihood_dem,
                                         likelihood_args=[flux,
                                                          err,
                                                          log_temp,
                                                          temp,
                                                          gofnt_matrix,
                                                          flux_weighting],
                                         n_walkers=n_walkers,
                                         burn_in_steps=burn_in_steps,
                                         production_steps=production_steps,
                                         thread_num=thread_num,
                                         count_num=count_num)
    np.save(OUTPUT_DIR+'samples_' + star_name +suffix, samples)
    np.save(OUTPUT_DIR+'lnprob_' + star_name +suffix, lnprob)
    return samples, lnprob, sampler


def generate_spectrum_data_npy(suffix='_q'):
    xray_fits = fits.open('../data_rsync/au_mic_data/au_mic_mos.ftz')
    arf_fits = fits.open('../data_rsync/au_mic_data/au_mic_mos_arf.ftz')

    xray_energy = (arf_fits[1].data['ENERG_LO'] + arf_fits[1].data['ENERG_HI'])
    xray_energy *= 0.5
    xray_wave = (xray_energy * u.keV).to(u.AA, equivalencies=u.spectral())
    xray_wave = xray_wave.value
    xray_counts = xray_fits[1].data['COUNTS']
    xray_counterr = np.sqrt(xray_counts)
    fix_mask = np.where(xray_counterr <= 0.1 * xray_counts)
    xray_counterr[fix_mask] = 0.1 * xray_counts[fix_mask]
    xray_photonflux = xray_counts / (xray_fits[1].header['EXPOSURE']
                                     * arf_fits[1].data['SPECRESP'])
    xray_photonerr = xray_counterr / (xray_fits[1].header['EXPOSURE']
                                      * arf_fits[1].data['SPECRESP'])

    photon_unit = u.photon / (u.s * (u.cm**2))
    flux_unit = u.erg / (u.s * (u.cm**2))

    xray_photonflux = (xray_photonflux * photon_unit)
    xray_photonerr = (xray_photonerr * photon_unit)
    spectral_equiv = u.spectral_density(xray_wave * u.AA)
    xray_flux = xray_photonflux.to(flux_unit, equivalencies=spectral_equiv)
    xray_flux = xray_flux.value
    xray_err = xray_photonerr.to(flux_unit, equivalencies=spectral_equiv)
    xray_err = xray_err.value

    xray_flux = xray_flux[np.argsort(xray_wave)]
    xray_err = xray_err[np.argsort(xray_wave)]
    xray_wave = np.sort(xray_wave)

    print('Loaded X-ray Data')

    mask = np.where((xray_wave >= 5.0) & (xray_wave <= 63.0)
                    & np.isfinite(xray_flux) & (np.isfinite(xray_err)))
    xray_flux = xray_flux[mask]
    xray_err = xray_err[mask]
    xray_wave = xray_wave[mask]
    xray_bins = np.diff(xray_wave)
    xray_bins = np.append(xray_bins[0], xray_bins)
    xray_flux /= xray_bins
    xray_err /= xray_bins
    _, xray_flux = bintoR(xray_wave, xray_flux,
                          R=15)
    xray_wave, xray_var = bintoR(xray_wave,
                                 xray_err**2, R=15)
    xray_err = np.sqrt(xray_var)

    print('Resampled X-ray data to R=15')

    xray_mask = np.where((np.isfinite(xray_flux)) &
                         (np.isfinite(xray_wave)) &
                         (np.isfinite(xray_err)) &
                         (xray_err > 0.0))
    xray_flux = xray_flux[xray_mask]
    xray_err = xray_err[xray_mask]
    xray_wave = xray_wave[xray_mask]
    xray_bins = np.diff(xray_wave)
    xray_bins = np.append(xray_bins[0], xray_bins)
    np.save('spectrum_data{}.npy'.format(suffix),
            [xray_wave, xray_bins,
             xray_flux, xray_err])
    return 'spectrum_data.npy'


if __name__ == '__main__':
    temp = np.logspace(4, 8, 100)
    log_temp = np.log10(temp)
    abund_type=15
    star_name_root = 'au_mic'
    abundance = 0.0
    star_rad = 0.75 * u.Rsun
    star_dist = 9.979 * u.pc
    flux_weighting = generate_flux_weighting(star_name_root,
                                             star_dist, star_rad)
    init_pos = [22.49331207, -3.31678227, -0.49848262,
                -1.27244452, -0.93897032, -0.67235648,
                -0.08085897]
    press_list = [1e17, 1e25, 1e24, 1e23, 1e22, 1e21, 1e20, 1e19, 1e18,
                  1e16, 1e15, 1e12, 1e13, 1e14]

    gofnt_dir = '/Users/arcticfox/Documents/AUMic/DEM/gofnt_dir'

    suffix = '_f'

    print('spectrum_data{}.npy'.format(suffix))
    if os.path.isfile(os.path.join(OUTPUT_DIR, 'spectrum_data{}.npy'.format(suffix))):
        data_npy_file = os.path.join(OUTPUT_DIR, 'spectrum_data{}.npy'.format(suffix))
    else:
        data_npy_file = generate_spectrum_data_npy()
        
    line_table_file = 'au_mic{0}_linetable.ascii'.format(suffix)
        
    for press in press_list:
        star_name = star_name_root + '_p' + str(int(np.log10(press)))
        gofnt_all = get_best_gofnt_matrix_press(abundance, press,
                                                    gofnt_dir=gofnt_dir)
        wave_all, bin_all = generate_constant_R_wave_arr(1, 1500, 100)
        
        if os.path.isfile(os.path.join(gofnt_dir, 'gofnt_' + star_name + suffix+'.npy')):
            gofnt_matrix = np.load(OUTPUT_DIR+'gofnt_' + star_name +suffix+ '.npy')
            flux = np.load(OUTPUT_DIR+'flux_' + star_name +suffix+ '.npy')
            err = np.load(OUTPUT_DIR+'err_' + star_name +suffix+ '.npy')
            
        else:
            print('main: ', gofnt_dir)
            out = get_star_data_gofnt_press(star_name,
                                            abundance,
                                        press,
                                            line_table_file,
                                            data_npy_file,
                                            abund_type=abund_type,
                                            gofnt_dir=gofnt_dir, suffix=suffix)
            gofnt_matrix, flux, err = out
            
        if os.path.isfile(OUTPUT_DIR+'samples_' + star_name + suffix + '.npy'):
            samples = np.load(OUTPUT_DIR+'samples_' + star_name +suffix+ '.npy')
            lnprob = np.load(OUTPUT_DIR+'lnprob_' + star_name +suffix+ '.npy')
        else:
            samples, lnprob, _ = run_mcmc_single_star(init_pos,
                                                      gofnt_matrix,
                                                      flux, err,
                                                      flux_weighting, suffix=suffix)
            
        if os.path.isfile(OUTPUT_DIR+'dem_' + star_name + suffix + '.pdf'):
            pass
        else:
            g = plot_dem(samples[:, :-1], lnprob, flux, gofnt_matrix,
                         log_temp, temp, flux_weighting,
                         'b', 'cornflowerblue', 0.1, 500,
                         'MCMC Samples', 'Best-fit DEM model',
                         r'$\Psi(T)$')
            g = display_fig(g, 'dem_' + star_name + suffix, mode='pdf')
            plt.clf()
            
        if os.path.isfile('corner_' + star_name + suffix + '.pdf'):
            pass
        else:
            h = corner.corner(samples, quantiles=[0.16, 0.5, 0.84],
                              show_titles=True, title_kwargs={"fontsize": 12},
                              plot_contours=True)
            h = display_fig(h, 'corner_' + star_name + suffix, mode='pdf')
            plt.clf()
        if os.path.isfile(OUTPUT_DIR+'spectrum_' + star_name + suffix + '.fits'):
            spectrum_table = Table.read(OUTPUT_DIR+'spectrum_' + star_name + suffix + '.fits')
            
        else:
            spectrum_name = OUTPUT_DIR + 'spectrum_' + star_name + suffix
            spectrum_table, _ = generate_spectrum_from_samples(spectrum_name,
                                                               samples,
                                                               lnprob,
                                                               gofnt_all,
                                                               flux_weighting,
                                                               wave_all,
                                                               bin_all)
        if os.path.isfile(OUTPUT_DIR+'spectrum_' + star_name + suffix + '.pdf'):
            pass
        else:
            spec_fig = plot_spectrum(OUTPUT_DIR+'spectrum_' + star_name + '.fits')
            spec_fig = display_fig(spec_fig, 'spectrum_' + star_name,
                                   'DEM Output Spectrum',
                                   mode='pdf')
