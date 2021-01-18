import numpy as np
import roman
from astropy.table import Table
from astropy import units as u
from typing import Tuple
from numpy.polynomial.chebyshev import chebval
from gofnt_routines import do_gofnt_matrix_integral


def parse_ascii_table_CHIANTI(fname):
    """Return table of ion fluxes with CHIANTI format.

    Parses an astropy.ascii table with the columns.
    ['Wavelength_vacuum', 'Ion', 'Flux', 'Error']
    and returns a table with the ion names in CHIANTI format.
    Assumes only a single ion per line without blends or molecules.

    Keyword arguments:
    :param fname: The filename of the ascii table to parse.
    :type fname: str

    Returns
    -------
    :return chianti_table: An Astropy table
    with the 'Ion' column
    rewritten to match CHIANTI format.

    """
    old_table = Table.read(fname, format='ascii')
    ion_list = old_table['Ion']
    split_ions = [ion_name.split() for ion_name in ion_list]
    new_names = [split_ion[0].lower() + '_' +
                 str(roman.fromRoman(split_ion[1][:-1]))
                 if ']' in split_ion[1] else split_ion[0].lower() + '_'
                 + str(roman.fromRoman(split_ion[1]))
                 for split_ion in split_ions]
    old_table['Ion'] = new_names
    return old_table


def generate_constant_R_wave_arr(start, end, R):
    min_wave = np.min([start, end])
    max_wave = np.max([start, end])
    min_dlambda = min_wave / R
    max_bins = (max_wave - min_wave) / min_dlambda
    wave_arr = np.empty((int(np.ceil(max_bins))))
    bin_arr = np.empty_like(wave_arr)
    temp_wave = min_wave
    i = 0
    while temp_wave <= max_wave:
        temp_dlambda = temp_wave / R
        bin_arr[i] = temp_dlambda
        wave_arr[i] = temp_wave + (0.5 * temp_dlambda)
        temp_wave += temp_dlambda
        i += 1
    return wave_arr[:i], bin_arr[:i]


def generate_constant_bin_wave_arr(start, end, bin):
    wave_arr = np.arange(start, end + bin, bin)
    bin_arr = np.ones_like(wave_arr) * bin
    return wave_arr, bin_arr


def get_bin_integral(wave_arr: np.ndarray, flux_arr: np.ndarray,
                     err_arr: np.ndarray,
                     bin_width: float = 3.0,
                     wave_min: float = 1230.0,
                     wave_max: float = 1602.0) -> Tuple[np.ndarray,
                                                        np.ndarray,
                                                        np.ndarray]:
    """Integrate the spectrum within bins of fixed width, mitigating dependence
    on instrument resolution and line shape to compare spectra (either observed
    or model generated).

    Keyword arguments:
    :param wave_arr: Wavelength array of original spectrum.
    :type wave_arr: np.ndarray.

    :param flux_arr: Flux array (in units of flux/wavelength) of original
    spectrum.
    :type flux_arr: np.ndarray.

    :param err_arr: Flux uncertainty array (in units of flux/wavelength)
    of original spectrum.
    :type err_arr: np.ndarray.

    :param bin_width: Size of wavelength bin to integrate spectrum within,
    in the same units as the original spectrum's wavelength array.
    :type bin_width: float.

    :param wave_min: Minimum wavelength for resampled bin-integrated flux
    :type wave_min: float.

    Returns:
    :returns: np.ndarray -- Wavelength array with bin centers.
    :returns: np.ndarray -- Integrated flux within each bin.
    :returns: np.ndarray -- Uncertainties of integrated flux within each bin

    """

    new_wave = np.arange(wave_min,
                         wave_max + bin_width, bin_width)
    new_flux = np.array(([np.trapz(
        flux_arr[np.where((wave_arr >= wave - 0.5 * bin_width) &
                          (wave_arr < wave + 0.5 * bin_width))],
        wave_arr[np.where((wave_arr >= wave - 0.5 * bin_width) &
                          (wave_arr < wave + 0.5 * bin_width))])
        for wave in new_wave]))
    old_var = err_arr**2.0
    new_var = np.array(([np.trapz(
        old_var[np.where((wave_arr >= wave - 0.5 * bin_width) &
                         (wave_arr < wave + 0.5 * bin_width))],
        wave_arr[np.where((wave_arr >= wave - 0.5 * bin_width) &
                          (wave_arr < wave + 0.5 * bin_width))])
        for wave in new_wave]))
    new_err = np.sqrt(new_var)
    err_mask = np.where(new_err == 0.0)
    new_err[err_mask] = np.inf
    return new_wave, new_flux, new_err


def do_threshold_mask(wave_arr, flux_arr, err_arr, threshold):
    mask = np.where(flux_arr > threshold)
    return wave_arr[mask], flux_arr[mask], err_arr[mask], mask


def generate_spectrum_from_samples(save_name, samples, lnprob,
                                   gofnt_spectrum, flux_weighting,
                                   spectrum_wave_arr, spectrum_bin_arr,
                                   sample_num=100000):
    temp = np.logspace(4, 8, 2000)
    log_temp = np.log10(temp)
    shift_log_temp = log_temp - np.mean(log_temp)
    range_temp = (np.max(shift_log_temp) - np.min(shift_log_temp))
    shift_log_temp /= (0.5 * range_temp)

    best_psi = 10.0**chebval(shift_log_temp,
                             samples[np.argmax(lnprob)][:-1])

    best_spectra2 = do_gofnt_matrix_integral(best_psi, gofnt_spectrum, temp,
                                             flux_weighting)
    best_spectra2 /= spectrum_bin_arr

    all_indices = np.arange(0, len(samples))
    rand_indices = np.random.choice(all_indices, sample_num)
    spec_len = len(spectrum_wave_arr)
    all_spectra = np.zeros((sample_num, spec_len))
    all_psi = np.zeros((sample_num, len(temp)))

    for i in range(sample_num):
        temp_psi = 10.0**(chebval(shift_log_temp,
                                  samples[rand_indices[i], :][:-1]))
        temp_spectra = do_gofnt_matrix_integral(temp_psi, gofnt_spectrum,
                                                temp, flux_weighting)
        temp_err = temp_spectra * (10.0**(samples[rand_indices[i]][-1]))
        all_spectra[i, :] = np.random.normal(loc=temp_spectra, scale=temp_err)
        all_spectra[i, :] /= spectrum_bin_arr
        all_psi[i, :] = temp_psi

    wave_unit = u.Angstrom
    flux_unit = u.erg / (u.s * u.cm**2)
    best_spectra = np.median(all_spectra, axis=0)

    med_psi = np.median(all_psi, axis=0)
    upp_psi = np.percentile(all_psi, 84, axis=0)
    low_psi = np.percentile(all_psi, 16, axis=0)

    upper_diff_var = (np.percentile(all_spectra, 84, axis=0) - best_spectra)**2
    lower_diff_var = (best_spectra - np.percentile(all_spectra, 16, axis=0))**2

    upper_var = upper_diff_var
    lower_var = lower_diff_var
    upper_err = np.sqrt(upper_var)
    lower_err = np.sqrt(lower_var)

    spectrum_table = Table([spectrum_wave_arr * wave_unit,
                            best_spectra * flux_unit,
                            lower_err * flux_unit,
                            upper_err * flux_unit,
                            best_spectra2 * flux_unit],
                           names=('Wavelength', 'Flux_density',
                                  'Lower_Error_16', 'Upper_Error_84',
                                  'Flux_density_ln_lmax'))
    spectrum_table.write(save_name + '.fits',
                         format='fits', overwrite=True)
    np.save(save_name + '_dems.npy', [low_psi, med_psi, upp_psi])
    return spectrum_table, best_spectra
