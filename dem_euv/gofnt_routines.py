import numpy as np
import ChiantiPy.core as ch
import roman
from astropy.table import Table
from ChiantiPy.tools.io import masterListRead
from typing import List, Any


def parse_ascii_table_CHIANTI(fname):
    """Parses an astropy.ascii table with the columns
    ['Wavelength_vacuum', 'Ion', 'Flux', 'Error']
    and returns a table with the ion names in CHIANTI format.
    Assumes only a single ion per line without blends or molecules.

    Keyword arguments:
    :param fname: The filename of the ascii table to parse.
    :type fname: str

    Returns:
    :return chianti_table: An Astropy table with the 'Ion' column
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


def generate_ion_gofnts(chianti_table, abundance=0.0, bin_width=1.0,
                        temp: np.ndarray = np.logspace(4.0, 8, 2000),
                        dens: np.ndarray = (1e17 / np.logspace(4.0, 8, 2000)),
                        abund_file='sun_coronal_2012_schmelz_ext'):
    """Given an Astropy table with columns
    ['Wavelength_vacuum', 'Ion', 'Flux', 'Error'] assuming the 'Ion' column
    matches CHIANTI format, for each unique ion returns:
    their emissivity functions, their summed flux, and their summed error.

    Keyword arguments:
    :param chianti_table: the astropy table as described above.
    :type chianti_table: astropy.table.Table

    :param temp: The array of temperature values to pass when creating the
    ChiantiPy ions. (default np.logspace(4.0, 8.0, 120))
    :type temp: np.ndarray.

    :param dens: The value of density to pass when creating the ChiantiPy ions.
    (default 10.0**11.0)
    :type dens: float.

    :param abund_file: The string with the name of the abundance file
    for ChiantiPy to query when creating the ions.
    (default sun_coronal_2012_schmelz_ext)
    :type abund_file: str.

    Returns:
    :returns ion_gofnts: np.array -- gofnt matrix of ion emissivities.
    :returns ion_fluxes: np.array -- list of fluxes for each ion
    :returns ion_errs: np.array -- list of errors on flux for each ion
    :returns ion_list: np.array -- list of ion names
    """
    ion_list = np.unique(chianti_table['Ion'])
    ion_gofnts = np.zeros((len(ion_list), len(temp)))
    ion_fluxes = np.zeros((len(ion_list)))
    ion_errs = np.zeros_like(ion_fluxes)
    for i in range(0, len(ion_list)):
        unique_ion = ion_list[i]
        ion_mask = np.where(chianti_table['Ion'] == unique_ion)
        temp_ion = ch.ion(unique_ion, temperature=temp,
                          eDensity=dens, abundance=abund_file)
        try:
            temp_ion.intensity()
            skip_ion = False
        except AttributeError:
            skip_ion = True
        if skip_ion:
            ion_gofnts[i, :] = np.inf
            ion_fluxes[i] = np.inf
            ion_errs[i] = np.inf
            ion_list[i] = 'SKIP'
        else:
            gofnt_prefactor = (temp_ion.Abundance * temp_ion.IoneqOne)
            gofnt_prefactor /= temp_ion.EDensity
            if temp_ion.Z > 2.0:
                gofnt_prefactor *= 10.0**abundance
            ion_fluxes[i] = np.sum(chianti_table['Flux'][ion_mask])
            ion_errs[i] = np.sqrt(
                np.sum((chianti_table['Error'][ion_mask])**2))
            done_waves = []
            all_waves = chianti_table['Wavelength_vacuum'][ion_mask]
            for j in range(0, len(all_waves)):
                wave_low = all_waves[j] - 0.5 * bin_width
                wave_high = all_waves[j] + 0.5 * bin_width
                skip_wave = False
                for done_wave in done_waves:
                    if all_waves[j] >= (done_wave - 0.5 * bin_width):
                        if all_waves[j] <= (done_wave + 0.5 * bin_width):
                            skip_wave = True
                if skip_wave:
                    pass
                else:
                    bin_mask = np.where(
                        (temp_ion.Emiss['wvl'] <= wave_high)
                        & (temp_ion.Emiss['wvl'] > wave_low))[0]
                    for line in bin_mask:
                        ion_gofnts[i, :] += gofnt_prefactor * \
                            temp_ion.Emiss['emiss'][line]
                done_waves.append(all_waves[j])
    good_mask = np.where(((np.mean(ion_gofnts, axis=1) > 0.0)
                          & (ion_list != 'SKIP')))[0]
    ion_gofnts = ion_gofnts[good_mask, :]
    ion_fluxes = ion_fluxes[good_mask]
    ion_errs = ion_errs[good_mask]
    ion_list = ion_list[good_mask]
    return ion_gofnts, ion_fluxes, ion_errs, ion_list


def initialize_ion(ion):
    ion.intensity()
    return ion


def get_gofnt_matrix_low_ram(ion_strs: List[Any],
                             wave_lows: np.ndarray, wave_upps: np.ndarray,
                             temp: np.ndarray, dens: np.ndarray,
                             abund_file: str = 'sun_coronal_2012_schmelz_ext',
                             abundance: float = 0.0) -> np.ndarray:
    """Create a matrix where the first axis is the array of wavelength bins
    within which to look for emission lines, while the second is
    the bin widths of the wavelength axis, and the third is the
    temperature array used for the ChiantiPy ions.
    The matrix contains contribution functions evaluated for all
    lines within a bin over the full temperature array.

    Keyword arguments:
    :param ions: List containing all ChiantiPy ions for which
    contribution functions should be evaluated.
    :type ions: list.

    :param wave_lows: Array of the wavelength bin lower bounds
    for which to evaluate contribution functions.
    :type wave_lows: np.ndarray.

    :param wave_upps: Array of the wavelength bin upper bounds
    for which to evaluate contribution functions.
    :type wave_upps: np.ndarray.

    :param temp: Array of temperatures for which
    the ChiantiPy emissivities have been evaluated.
    :type temp: np.ndarray.

    :param abundance: [Fe/H] abundance to weight the contribution functions by.
    (default 0.0)
    :type abundance: float.

    Returns:
    :returns: np.ndarray -- Contribution function matrix
    divided by wavelength bin width.

    """
    gofnt_matrix = np.zeros((len(wave_lows), len(temp)))
    bin_arr = wave_upps - wave_lows
    wave_arr = wave_lows + (0.5 * bin_arr)

    for ion_str in ion_strs:
        print(ion_str)
        ion = initialize_ion(ch.ion(ion_str, temperature=temp,
                                    eDensity=dens, abundance=abund_file))
        gofnt_prefactor = ion.Abundance * ion.IoneqOne / ion.EDensity
        if ion.Z > 2.0:
            gofnt_prefactor *= 10.0**abundance
        for i in range(0, len(wave_arr)):
            wave_low = wave_arr[i] - 0.5 * bin_arr[i]
            wave_high = wave_arr[i] + 0.5 * bin_arr[i]
            bin_mask = np.where((ion.Emiss['wvl'] <= wave_high) & (
                ion.Emiss['wvl'] > wave_low))[0]
            for line in bin_mask:
                gofnt_matrix[i, :] += gofnt_prefactor * \
                    ion.Emiss['emiss'][line]

    for ion_str in ['h_1', 'h_2', 'he_1', 'he_2', 'he_3']:
        print(ion_str)
        if ion_str in ['h_2', 'he_3']:
            pass
        else:
            ion = initialize_ion(ch.ion(ion_str, temperature=temp,
                                        eDensity=dens, abundance=abund_file))
            ion.twoPhoton(wave_arr)
            if 'intensity' in ion.TwoPhoton.keys():
                twophot_contrib = ion.TwoPhoton['intensity'].T
                twophot_contrib *= bin_arr.reshape((bin_arr.size, 1))
                tp_mask = np.where(np.isfinite(twophot_contrib))
                gofnt_matrix[tp_mask] += twophot_contrib[tp_mask]
            else:
                print('2Photon failed for', ion_str)
        if ion_str in ['h_2', 'he_2', 'he_3']:
            cont = ch.continuum(ion_str, temp, abundance=abund_file)
            cont.freeFree(wave_arr)
            cont.freeBound(wave_arr)
            if 'intensity' in cont.FreeFree.keys():
                freefree_contrib = cont.FreeFree['intensity'].T
                freefree_contrib *= bin_arr.reshape((bin_arr.size, 1))
                ff_mask = np.where(np.isfinite(freefree_contrib))
                gofnt_matrix[ff_mask] += freefree_contrib[ff_mask]
            else:
                print('No FreeFree intensity calculated for ',  ion_str)
            if 'intensity' in cont.FreeBound.keys():
                freebound_contrib = cont.FreeBound['intensity'].T
                freebound_contrib *= bin_arr.reshape((bin_arr.size, 1))
                fb_mask = np.where(np.isfinite(freebound_contrib))
                gofnt_matrix[fb_mask] += freebound_contrib[fb_mask]
            else:
                print('No FreeBound intensity calculated for ', ion_str)
        else:
            pass
    return gofnt_matrix


def do_gofnt_matrix_integral(psi_model: np.ndarray,
                             gofnt_matrix: np.ndarray,
                             temp: np.ndarray,
                             flux_weighting: float) -> np.ndarray:
    """Evaluate the line intensity integrals for each wavelength bin using the
    contribution function matrix and a given DEM model.

    Keyword arguments:
    :param psi_model: DEM model for the integral
    :type psi_model: np.ndarray.

    :param gofnt_matrix: Contribution function matrix
    :type gofnt_matrix: np.ndarray.

    :param temp: Array of temperatures along same axis as the DEM model, the
    ChiantiPy emissivities, and the second axis of the contribution function
    matrix.
    :type temp: np.ndarray.

    :param flux_weighting: Weight the intensity integral to map to
    an observable or desired quantity
    (flux received by exoplanet, surface flux of star etc.)
    :type flux_weighting: float.

    Returns:
    :returns: np.ndarray -- Total flux intensities in each wavelength bin.

    """

    integrated_intensity_array = np.zeros((np.shape(gofnt_matrix)[0]))
    for i in np.arange(0, len(integrated_intensity_array)):
        integrated_intensity_array[i] = np.trapz(
            gofnt_matrix[i, :] * psi_model, temp) * (flux_weighting)
    return integrated_intensity_array


def resample_gofnt_matrix(gofnt_old, wave_new, bin_new, wave_old):
    gofnt_new = np.zeros((len(wave_new), np.shape(gofnt_old)[1]))
    wave_highs = wave_new + (0.5 * bin_new)
    wave_lows = wave_new - (0.5 * bin_new)
    for i in range(len(wave_new)):
        wave_mask = np.where((wave_old <= wave_highs[i]) &
                             (wave_old > wave_lows[i]))
        gofnt_new[i] += (np.sum((gofnt_old[wave_mask[0], :]), axis=0))
    return gofnt_new
