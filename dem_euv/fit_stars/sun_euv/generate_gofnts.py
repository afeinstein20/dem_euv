import os
import roman
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ChiantiPy.core as ch
from gofnt_routines import get_gofnt_matrix_low_ram, initialize_ion
from gofnt_routines import masterListRead
from data_prep import generate_constant_R_wave_arr


sns.set_context('paper')
sns.set_style('ticks')
plt.rc('font', family='serif')
plt.rc('text', usetex=True)


def generate_gofnts_by_abundance(abundances, abund_strs, ions,
                                 wave_arr, bin_arr,
                                 wave_min, wave_max,
                                 logtemp_min, logtemp_max,
                                 const_R, temp, dens, abund_file,
                                 logtemp_min_str, logtemp_max_str, press,
                                 save_loc=''):
    gofnt_str = 'gofnt_'
    gofnt_str += 'w' + str(int(wave_min)) + '_'
    gofnt_str += 'w' + str(int(wave_max))
    gofnt_str += 't' + logtemp_min_str + '_'
    gofnt_str += 't' + logtemp_max_str + '_'
    gofnt_str += 'r' + str(int(const_R)) + '_'
    gofnt_str += 'p' + str(int(np.log10(press))) + '_'
    if save_loc != '':
        os.makedirs(save_loc, exist_ok=True)
    wave_lows = wave_arr - (0.5 * bin_arr)
    wave_upps = wave_lows + bin_arr
    for abundance, abund_str in zip(abundances, abund_strs):
        temp_str = gofnt_str + abund_str + '.npy'
        print('Generating: ', temp_str)
        gofnt = get_gofnt_matrix_low_ram(ions, wave_lows, wave_upps,
                                         temp, dens, abund_file, abundance)
        np.save(save_loc + temp_str, gofnt)


def visualize_gofnt(gofnt_matrix, wave_arr, logtemp_arr,
                    vmin, vmax):
    fig = plt.figure(1)
    pcmesh = plt.pcolormesh(logtemp_arr, wave_arr, gofnt_matrix,
                            cmap='inferno',
                            vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlabel(r'$\log_{10} (T \, \, [\mathrm{K}]) $')
    plt.ylabel(r'Wavelength [$\mathrm{\AA}$]')
    plt.tight_layout()
    return fig, pcmesh


def gofnt_heatmap_plot():
    plt.savefig('gofnt_heatmap.png', dpi=1000)
    plt.savefig('gofnt_heatmap.pdf')
    plt.show()


def get_ion_gofnt_matrices(ion_strs,
                           wave_arr, bin_arr,
                           temp, dens,
                           abund_file='sun_coronal_2012_schmelz_ext',
                           abundance=0.0, vmin=8.359e-27, vmax=3.5963e-27):
    """

    Keyword arguments:
    :param ions: List containing all ChiantiPy ions for which
    contribution functions should be evaluated.
    :type ions: list.

    :param wave_arr: Array of the wavelength bin centers for which to evaluate
    contribution functions.
    :type wave_arr: np.ndarray.

    :param bin_arr: Array of the wavelength bin widths for which to evaluate
    and weight the contribution functions.
    :type wave_arr: np.ndarray.

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
    dens_str = 'd' + str(int(np.log10(dens)))
    temp_str = 't' + str(int(np.log10(temp[0]))) + '_'
    temp_str += str(int(np.log10(temp[-1])))
    abundance_str = 'a' + str(int(abundance)) + '_' + abund_file
    base_dir_str = 'abundance_' + abundance_str
    base_dir_str += '_' + temp_str
    base_dir_str += '_' + dens_str
    plot_dir = base_dir_str + '/gofnt_plots'
    if os.path.exists(base_dir_str):
        pass
    else:
        os.mkdir(base_dir_str)
    if os.path.exists(plot_dir):
        pass
    else:
        os.mkdir(plot_dir)
    for ion_str in ion_strs:
        ion_dir_str = base_dir_str + '/' + ion_str
        ion_gofnt_str = ion_dir_str + '/' + ion_str + '_gofnt.npy'
        ion_gofnt_plot_str = plot_dir + '/' + ion_str + '.png'
        if os.path.exists(ion_dir_str):
            pass
        else:
            os.mkdir(ion_dir_str)

        if os.path.isfile(ion_gofnt_str):
            pass
        else:
            ion_gofnt_matrix = np.zeros((len(wave_arr), len(temp)))
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
                    ion_gofnt_matrix[i, :] += gofnt_prefactor * \
                        ion.Emiss['emiss'][line]
            np.save(ion_gofnt_str, ion_gofnt_matrix)
        if os.path.isfile(ion_gofnt_plot_str):
            pass
        else:
            ion_gofnt_matrix = np.load(ion_gofnt_str)
            fig, pcmesh = visualize_gofnt(ion_gofnt_matrix,
                                          wave_arr, np.log10(temp),
                                          vmin, vmax)
            ion_split = ion_str.split('_')
            title_str = ion_split[0].capitalize()
            if ion_split[1].endswith('d'):
                title_str += ' ' + roman.toRoman(int(ion_split[1][:-1])) + 'd'
            else:
                title_str += ' ' + roman.toRoman(int(ion_split[1]))
            title_str += r' [erg cm$^{3}$ s$^{-1}$ sr$^{-1}$]'
            plt.title(title_str)
            plt.savefig(ion_gofnt_plot_str, dpi=500)
            plt.clf()
    return None


def generate_standard_gofnt_library_pressure(save_loc=''):
    wave_min = 1.0
    wave_max = 1500.0

    const_R = 100

    logtemp_min = 4.0
    logtemp_max = 8
    n_points = 2000

    press_list = [1e15, 1e17, 1e16, 1e18, 1e19, 1e20, 1e13, 1e14, 1e15]
    temp = np.logspace(logtemp_min, logtemp_max, n_points)
    abund_file = 'sun_coronal_2012_schmelz_ext'

    wave_arr, bin_arr = generate_constant_R_wave_arr(wave_min, wave_max,
                                                     const_R)

    abundances = [0.0, -1.0, 1.0]
    abund_strs = ['sol0', 'sub1', 'sup1']
    for press in press_list:
        dens = press / temp
        ions = masterListRead()
        print(ions)
        generate_gofnts_by_abundance(abundances, abund_strs, ions,
                                     wave_arr, bin_arr,
                                     wave_min, wave_max,
                                     logtemp_min, logtemp_max,
                                     const_R, temp, dens, abund_file,
                                     '4', '8', press, save_loc)


def generate_specific_gofnt_library_pressure(star_name,
                                             abund_file, press_list,
                                             save_loc='', abundance=0.0):
    wave_min = 1.0
    wave_max = 1500.0

    const_R = 100

    logtemp_min = 4.0
    logtemp_max = 8
    n_points = 2000
    temp = np.logspace(logtemp_min, logtemp_max, n_points)

    wave_arr, bin_arr = generate_constant_R_wave_arr(wave_min, wave_max,
                                                     const_R)
    gofnt_str = 'gofnt_'
    gofnt_str += 'w' + str(int(wave_min)) + '_'
    gofnt_str += 'w' + str(int(wave_max)) + '_'
    gofnt_str += 't' + str(int(logtemp_min)) + '_'
    gofnt_str += 't' + str(int(logtemp_max)) + '_'
    gofnt_str += 'r' + str(int(const_R)) + '_'
    if save_loc != '':
        os.makedirs(save_loc, exist_ok=True)
    wave_lows = wave_arr - (0.5 * bin_arr)
    wave_upps = wave_lows + bin_arr
    for press in press_list:
        dens = press / temp
        ions = masterListRead()
        temp_str = gofnt_str + 'p' + str(int(np.log10(press))) + '_'
        temp_str += star_name + '.npy'
        print(ions)
        print('Generating: ', temp_str)
        gofnt = get_gofnt_matrix_low_ram(ions, wave_lows, wave_upps,
                                         temp, dens, abund_file, abundance)
        np.save(save_loc + temp_str, gofnt)


if __name__ == '__main__':
    save_loc = '../../gofnt_dir/'
    press_list_1 = [1e17, 1e15]
    press_list_2 = [1e13, 1e14, 1e19, 1e16, 1e18, 1e20, 1e21, 1e22]
    abund_file_au_mic = 'unity'
    abund_file_sol = 'sun_coronal_2012_schmelz_ext'
    generate_specific_gofnt_library_pressure('sol0',
                                             abund_file_sol,
                                             press_list_1, save_loc, 0.0)
    generate_specific_gofnt_library_pressure('sol0',
                                             abund_file_sol,
                                             press_list_2, save_loc, 0.0)
    generate_specific_gofnt_library_pressure('sup1',
                                             abund_file_sol,
                                             press_list_2, save_loc, 1.0)
    generate_specific_gofnt_library_pressure('sub1',
                                             abund_file_sol,
                                             press_list_2, save_loc, -1.0)
    generate_specific_gofnt_library_pressure('sup1',
                                             abund_file_sol,
                                             press_list_1, save_loc, 1.0)
    generate_specific_gofnt_library_pressure('sub1',
                                             abund_file_sol,
                                             press_list_1, save_loc, -1.0)
