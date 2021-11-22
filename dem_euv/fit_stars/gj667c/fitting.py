import numpy as np
import emcee
import multiprocessing
from numpy.polynomial.chebyshev import chebval
from typing import List, Callable, Any, Union
from gofnt_routines import do_gofnt_matrix_integral


def ln_prior_cutoff_dem(params: List[float],
                        psi_low: float = 17.0,
                        psi_high: float = 29.0) -> float:
    """Apply a uniform prior between 10**+/- 2 for coefficients used in
    Chebyshev polynomial model for DEM. The c_0 coefficient is sampled
    uniformly between two limits and then gets an exponential penalty
    such that the probability is 1/e less likely at limits psi_low and psi_high
    which are set by reasonable physical assumptions for the DEM.
    The temperature interval is 1e8 K, the path-length varies between
    0.01 R_sun and 10 R_sun (1e8 and 1e11 cm), and the n_e varies between
    1e8 and 1e13 cm^-3

    Keyword arguments:
    :param params: List of coefficients for Chebyshev polynomial in format used
    by numpy chebyshev polynomial followed by the predicted flux factor
    uncertainty.
    :type params: list.

    Returns:
    :returns: float -- 0 if prior is satisfied, -np.inf otherwise.

    """
    flux_factor = params[-1]
    coeffs = params[:-1]
    lp = 0.0
    if coeffs[0] >= psi_high:
        lp += psi_high - coeffs[0]
    elif coeffs[0] <= psi_low:
        lp += coeffs[0] - psi_low
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


def ln_prob_flux_sigma_dem(params: List[float], y: np.ndarray,
                           yerr: Union[np.ndarray, float],
                           log_temp: np.ndarray, temp: np.ndarray,
                           gofnt_matrix: np.ndarray,
                           flux_weighting: float) -> float:
    """Evaluate the likelihood with an additional variance term associated
    with the uncertainty of the predicted flux from the model
    Keyword arguments:
    :param params: List of coefficients for Chebyshev polynomial in format used
    by numpy chebyshev polynomial followed by the predicted flux factor
    uncertainty.
    :type params: list.

    :param y: Bin-integral of observed spectrum.
    :type y: np.ndarray.

    :param yerr: Error on y, either an array or constant float.
    :type yerr: Union[np.ndarray, float].

    :param log_temp: log10(Temperature  array for ChiantiPy emissivities)
    :type log_temp: np.ndarray.

    :param temp: Temperature array for ChiantiPy emissivities.
    :type temp: np.ndarray.

    :param gofnt_matrix: Contribution matrix along wavelength bin and
    temperature arrays.
    :type gofnt_matrix: np.ndarray.

    :param flux_weighting: Weight the intensity integral to map to
    some observable or desired quantity
    (flux received by exoplanet, surface flux of star etc.)
    :type flux_weighting: float.

    Returns:
    :returns: float -- -1/2 Chi-squared by comparing bin integral of observed
    spectrum to DEM integrated spectrum

    """
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


def ln_likelihood_dem(params: List[float],
                      y: np.ndarray,
                      yerr: Union[np.ndarray, float],
                      log_temp: np.ndarray, temp: np.ndarray,
                      gofnt_matrix: np.ndarray,
                      flux_weighting: float,
                      ln_prob_func: Callable = ln_prob_flux_sigma_dem,
                      ln_prior_func: Callable = ln_prior_cutoff_dem) -> float:
    """Combine a defined prior and probability function to determine the
    ln_likelihood of a model given two types of data: the bin-integral of an
    observed spectrum or the intensities of individual lines.

    Keyword arguments:
    :param params: List of coefficients for Chebyshev polynomial in format used
    by numpy chebyshev polynomial followed by the predicted flux factor
    uncertainty.
    :type params: list.

    :param y: Bin-integral of observed spectrum.
    :type y: np.ndarray.

    :param yerr: Error on y, either an array or constant float.
    :type yerr: Union[np.ndarray, float].

    :param log_temp: log10(Temperature  array for ChiantiPy emissivities)
    :type log_temp: np.ndarray.

    :param temp: Temperature array for ChiantiPy emissivities.
    :type temp: np.ndarray.

    :param wave_arr: Wavelength array with bin centers.
    :type wave_arr: np.ndarray.

    :param gofnt_matrix: Contribution matrix along wavelength bin and
    temperature arrays.
    :type gofnt_matrix: np.ndarray.

    :param flux_weighting: Weight the intensity integral to map to
    some observable or desired quantity
    (flux received by exoplanet, surface flux of star etc.)
    :type flux_weighting: float.

    Returns:
    :returns: float -- ln_likelihood from comparing bin integral of observed
    spectrum to DEM integrated spectrum or individual line intensities to DEM
    integrated line intensities.

    """

    lp = ln_prior_func(params)
    if np.isfinite(lp):
        return lp + ln_prob_func(params, y, yerr, log_temp, temp,
                                 gofnt_matrix, flux_weighting)
    return -np.inf


def fit_emcee(init_pos: np.ndarray,
              likelihood_func: Callable,
              likelihood_args: List[Any],
              n_walkers: int,
              burn_in_steps: int,
              production_steps: int,
              init_spread: float = 1e-3,
              second_spread: float = 1e-4,
              double_burn: bool = True,
              thread_num: int = multiprocessing.cpu_count(),
              count_print: bool = True,
              count_num: int = 100):
    """Run the emcee sampler with a given likelihood function
    and return the flatchain samples, flatchain ln_probability, and the sampler
    object.

    Keyword arguments:
    :param init_pos: Initial values for the model parameters.
    : type init_pos: np.ndarray.

    :param likelihood_func: Likelihood function for the model.
    :type likelihood_func: function.

    :param likelihood_args: Arguments required for the likelihood function.
    :type likelihood_args: list.

    :param zxn_walkers: Number of walkers for the emcee sampler.
    :type n_walkers: int.

    :param burn_in_steps: Number of steps for the burn-in phase.
    :type burn_in_steps: int.

    :param production_steps: Number of steps for the production phase.
    :type production_steps: int.

    :param init_spread: Multiplicative factor by which to scramble the initial
    position of the walkers. (default 1e-3)
    :type init_spread: float.

    :param second_spread: Multiplicative factor by which to scramble the
    highest likelihood position after the burn-in phase. (default 1e-4)
    :type second_spread: float.

    :param double_burn: Whether or not to do a second burn-in phase. Treated
    identically to the initial. (default True)
    :type double_burn: bool

    :param thread_num: Number of threads for the emcee sampler to use.
    (default cpu_count)
    :type thread_num: int.

    :param count_print: Whether or not to print progress messages during
    production.
    :type count_print: bool.

    :param count_num: Interval for print messages.
    :type count_num: int.

    Returns:
    :returns: np.ndarray -- Reshaped sampler positions, collapsed along walker
    axis (see emcee documentation).

    :returns: np.ndarray -- ln_probability values for walker positions aligned
    to the flatchain.

    :returns: emcee.sampler -- emcee sampler object, refer to emcee
    documentation.

    """

    ndim = len(init_pos)
    pos = [init_pos + init_spread * np.random.randn(ndim) * init_pos
           for i in range(n_walkers)]
    print("Initializing walkers")
    sampler = emcee.EnsembleSampler(n_walkers, ndim, likelihood_func,
                                    args=likelihood_args, threads=thread_num)
    print("Starting burn-in")
    p0, prob, _ = sampler.run_mcmc(pos, burn_in_steps)
    p0 = [p0[np.argmax(prob)] + second_spread * np.random.randn(ndim) *
          p0[np.argmax(prob)] for i in range(n_walkers)]
    sampler.reset()
    nsteps = burn_in_steps + production_steps
    done_steps = burn_in_steps
    if double_burn:
        print('Starting second burn-in')
        nsteps += burn_in_steps
        done_steps += burn_in_steps
        p0, prob, _ = sampler.run_mcmc(p0, burn_in_steps)
        p0 = [p0[np.argmax(prob)] + second_spread * np.random.randn(ndim) *
              p0[np.argmax(prob)] for i in range(n_walkers)]
        sampler.reset()
    print('Starting production')
    if count_print:
        for i, _ in enumerate(sampler.sample(p0, iterations=production_steps)):
            if (i + 1) % count_num == 0:
                print("{0:5.1%}".format(float(i + done_steps) / nsteps))
    else:
        p0, prob, _ = sampler.run_mcmc(p0, production_steps)
    return sampler.flatchain, sampler.flatlnprobability, sampler
