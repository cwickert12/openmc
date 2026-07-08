from numbers import Real
from math import exp, erf, pi, sqrt
from copy import deepcopy

import os
import h5py
import pickle
import numpy as np
from scipy.signal import find_peaks

import openmc.checkvalue as cv
from openmc.exceptions import DataError
from openmc.mixin import EqualityMixin
from openmc.data import WMP_VERSION, WMP_VERSION_MAJOR
from openmc.data.data import K_BOLTZMANN
from openmc.data.neutron import IncidentNeutron
from openmc.data.resonance import ResonanceRange


# Index of the pole in each row of the data array
_MP_EA = 0       # Pole

# Upper temperature limit (K)
TEMPERATURE_LIMIT = 3000

# Logging control
DETAILED_LOGGING = 2


def _faddeeva(z):
    r"""Evaluate the complex Faddeeva function.

    Technically, the value we want is given by the equation:

    .. math::
        w(z) = \frac{i}{\pi} \int_{-\infty}^{\infty} \frac{1}{z - t}
        \exp(-t^2) \text{d}t

    as shown in Equation 63 from Hwang, R. N. "A rigorous pole
    representation of multilevel cross sections and its practical
    applications." Nuclear Science and Engineering 96.3 (1987): 192-209.

    The :func:`scipy.special.wofz` function evaluates
    :math:`w(z) = \exp(-z^2) \text{erfc}(-iz)`. These two forms of the Faddeeva
    function are related by a transformation.

    If we call the integral form :math:`w_\text{int}`, and the function form
    :math:`w_\text{fun}`:

    .. math::
        w_\text{int}(z) =
        \begin{cases}
            w_\text{fun}(z) & \text{for } \text{Im}(z) > 0\\
            -w_\text{fun}(z^*)^* & \text{for } \text{Im}(z) < 0
        \end{cases}

    Parameters
    ----------
    z : complex
        Argument to the Faddeeva function.

    Returns
    -------
    complex
        :math:`\frac{i}{\pi} \int_{-\infty}^{\infty} \frac{1}{z - t} \exp(-t^2)
        \text{d}t`

    """
    from scipy.special import wofz
    if np.angle(z) > 0:
        return wofz(z)
    else:
        return -np.conj(wofz(z.conjugate()))


def _broaden_wmp_polynomials(E, dopp, n):
    r"""Evaluate Doppler-broadened windowed multipole curvefit.

    The curvefit is a polynomial of the form :math:`\frac{a}{E}
    + \frac{b}{\sqrt{E}} + c + d \sqrt{E} + \ldots`

    Parameters
    ----------
    E : float
        Energy to evaluate at.
    dopp : float
        sqrt(atomic weight ratio / kT) in units of eV.
    n : int
        Number of components to the polynomial.

    Returns
    -------
    np.ndarray
        The value of each Doppler-broadened curvefit polynomial term.

    """
    sqrtE = sqrt(E)
    beta = sqrtE * dopp
    half_inv_dopp2 = 0.5 / dopp**2
    quarter_inv_dopp4 = half_inv_dopp2**2

    if beta > 6.0:
        # Save time, ERF(6) is 1 to machine precision.
        # beta/sqrtpi*exp(-beta**2) is also approximately 1 machine epsilon.
        erf_beta = 1.0
        exp_m_beta2 = 0.0
    else:
        erf_beta = erf(beta)
        exp_m_beta2 = exp(-beta**2)

    # Assume that, for sure, we'll use a second order (1/E, 1/V, const)
    # fit, and no less.

    factors = np.zeros(n)

    factors[0] = erf_beta / E
    factors[1] = 1.0 / sqrtE
    factors[2] = (factors[0] * (half_inv_dopp2 + E)
                  + exp_m_beta2 / (beta * sqrt(pi)))

    # Perform recursive broadening of high order components. range(1, n-2)
    # replaces a do i = 1, n-3.  All indices are reduced by one due to the
    # 1-based vs. 0-based indexing.
    for i in range(1, n-2):
        if i != 1:
            factors[i+2] = (-factors[i-2] * (i - 1.0) * i * quarter_inv_dopp4
                + factors[i] * (E + (1.0 + 2.0 * i) * half_inv_dopp2))
        else:
            factors[i+2] = factors[i]*(E + (1.0 + 2.0 * i) * half_inv_dopp2)

    return factors


def _vectfit_xs(energy, ce_xs, mts, rtol=1e-3, atol=1e-5, orders=None,
                n_vf_iter=30, log=False, path_out=None, poly_orders=0):
    """Convert point-wise cross section to multipole data via vector fitting.

    Parameters
    ----------
    energy : np.ndarray
        Energy array
    ce_xs : np.ndarray
        Point-wise cross sections to be fitted, with shape (number of reactions,
        number of energy points)
    mts : Iterable of int
        Reaction list
    rtol : float, optional
        Relative error tolerance
    atol : float, optional
        Absolute error tolerance
    orders : Iterable of int, optional
        A list of orders (number of poles) to be searched
    n_vf_iter : int, optional
        Number of maximum VF iterations
    log : bool or int, optional
        Whether to print running logs (use int for verbosity control)
    path_out : str, optional
        Path to save the figures to show discrepancies between the original and
        fitted cross sections for different reactions

    Returns
    -------
    tuple
        (poles, residues)

    """
    
    # import vectfit package: https://github.com/liangjg/vectfit
    import vectfit as vf

    ne = energy.size
    nmt = len(mts)
    if ce_xs.shape != (nmt, ne):
        raise ValueError('Inconsistent cross section data.')

    # construct test data: interpolate xs with finer grids
    n_finer = 10
    ne_test = (ne - 1)*n_finer + 1
    test_energy = np.interp(np.arange(ne_test),
                            np.arange(ne_test, step=n_finer), energy)
    test_energy[[0, -1]] = energy[[0, -1]]  # avoid numerical issue
    test_xs_ref = np.zeros((nmt, ne_test))
    for i in range(nmt):
        test_xs_ref[i] = np.interp(test_energy, energy, ce_xs[i])

    if log:
        print(f"  energy: {energy[0]:.3e} to {energy[-1]:.3e} eV ({ne} points)")
        print(f"  error tolerance: rtol={rtol}, atol={atol}")

    # transform xs (sigma) and energy (E) to f (sigma*E) and s (sqrt(E)) to be
    # compatible with the multipole representation
    f = ce_xs * energy
    s = np.sqrt(energy)
    test_s = np.sqrt(test_energy)

    # inverse weighting is used for minimizing the relative deviation instead of
    # absolute deviation in vector fitting
    with np.errstate(divide='ignore'):
        weight = 1.0/np.abs(f) # cwickert change to absolute value to try and fix the problem with the higher order moments

    # avoid too large weights which will harm the fitting accuracy
    min_cross_section = 1e-7
    for i in range(nmt):
        if np.all(ce_xs[i] <= min_cross_section):
            weight[i] = 1.0
        elif np.any(ce_xs[i] <= min_cross_section):
            weight[i, ce_xs[i] <= min_cross_section] = \
               max(weight[i, ce_xs[i] > min_cross_section])

    # detect peaks (resonances) and determine VF order search range
    peaks, _ = find_peaks(ce_xs[0] + ce_xs[1])
    n_peaks = peaks.size
    if orders is not None:
        # make sure orders are even integers
        orders = list(set([int(i/2)*2 for i in orders if i >= 2]))
    else:
        lowest_order = max(2, 2*n_peaks)
        highest_order = max(200, 4*n_peaks)
        orders = list(range(lowest_order, highest_order + 1, 2))

    if log:
        print(f"Found {n_peaks} peaks")
        print(f"Fitting orders from {orders[0]} to {orders[-1]}")

    # perform VF with increasing orders
    found_ideal = False
    n_discarded = 0  # for accelation, number of discarded searches
    best_quality = best_ratio = -np.inf
    for i, order in enumerate(orders):
        if log:
            print(f"Order={order}({i}/{len(orders)})")
        # initial guessed poles
        poles_r = np.linspace(s[0], s[-1], order//2)
        poles = poles_r + poles_r*0.01j
        poles = np.sort(np.append(poles, np.conj(poles)))

        found_better = False
        # fitting iteration
        for i_vf in range(n_vf_iter):
            if log >= DETAILED_LOGGING:
                print(f"VF iteration {i_vf + 1}/{n_vf_iter}")

            # call vf
            poles, residues, cf, f_fit, rms = vf.vectfit(f, s, poles, weight, n_polys=poly_orders)

            # convert real pole to conjugate pairs
            n_real_poles = 0
            new_poles = []
            for p in poles:
                p_r, p_i = np.real(p), np.imag(p)
                if (s[0] <= p_r <= s[-1]) and p_i == 0.:
                    new_poles += [p_r+p_r*0.01j, p_r-p_r*0.01j]
                    n_real_poles += 1
                else:
                    new_poles += [p]
            new_poles = np.array(new_poles)
            # re-calculate residues if poles changed
            if n_real_poles > 0:
                if log >= DETAILED_LOGGING:
                    print(f"  # real poles: {n_real_poles}")
                new_poles, residues, cf, f_fit, rms = \
                      vf.vectfit(f, s, new_poles, weight, skip_pole=True, n_polys=poly_orders)

            # assess the result on test grid
            test_xs = vf.evaluate(test_s, new_poles, residues) / test_energy
            abserr = np.abs(test_xs - test_xs_ref)
            with np.errstate(invalid='ignore', divide='ignore'):
                relerr = abserr / test_xs_ref
                if np.any(np.isnan(abserr)):
                    maxre, ratio, ratio2 = np.inf, -np.inf, -np.inf
                elif np.all(abserr <= atol):
                    maxre, ratio, ratio2 = 0., 1., 1.
                else:
                    maxre = np.max(relerr[abserr > atol])
                    ratio = np.sum((relerr < rtol) | (abserr < atol)) / relerr.size
                    ratio2 = np.sum((relerr < 10*rtol) | (abserr < atol)) / relerr.size

            # define a metric for choosing the best fitting results
            # basically, it is preferred to have more points within accuracy
            # tolerance, smaller maximum deviation and fewer poles
            #TODO: improve the metric with clearer basis
            quality = ratio + ratio2 - min(0.1*maxre, 1) - 0.001*new_poles.size

            #if np.any(test_xs < -atol): # comment out for now since higher order moments can be negative
            #    quality = -np.inf

            if log >= DETAILED_LOGGING:
                print(f"  # poles: {new_poles.size}")
                print(f"  Max relative error: {maxre * 100:.3f}%")
                print(f"  Satisfaction: {ratio * 100:.1f}%, {ratio2 * 100:.1f}%")
                print(f"  Quality: {quality:.2f}")

            if quality > best_quality:
                if log >= DETAILED_LOGGING:
                    print("  Best so far!")
                found_better = True
                best_quality, best_ratio = quality, ratio
                best_poles, best_residues = new_poles, residues
                best_test_xs, best_relerr = test_xs, relerr
                if best_ratio >= 1.0:
                    if log:
                        print("Found ideal results. Stop!")
                    found_ideal = True
                    break
            else:
                if log >= DETAILED_LOGGING:
                    print("  Discarded!")

        if found_ideal:
            break

        # acceleration
        if found_better:
            n_discarded = 0
        else:
            if order > max(2*n_peaks, 50) and best_ratio > 0.7:
                n_discarded += 1
                if n_discarded >= 10 or (n_discarded >= 5 and best_ratio > 0.9):
                    if log >= DETAILED_LOGGING:
                        print("Couldn't get better results. Stop!")
                    break

    # merge conjugate poles
    real_idx = []
    conj_idx = []
    found_conj = False
    for i, p in enumerate(best_poles):
        if found_conj:
            found_conj = False
            continue
        if np.imag(p) == 0.:
            real_idx.append(i)
        else:
            if i < best_poles.size and np.conj(p) == best_poles[i + 1]:
                found_conj = True
                conj_idx.append(i)
            else:
                raise RuntimeError("Complex poles are not conjugate!")
    if log:
        print("Found {} real poles and {} conjugate complex pairs.".format(
               len(real_idx), len(conj_idx)))
    mp_poles = best_poles[real_idx + conj_idx]
    mp_residues = np.concatenate((best_residues[:, real_idx],
                                  best_residues[:, conj_idx]*2), axis=1)/1j
    if log:
        print(f"Final number of poles: {mp_poles.size}")

    if path_out:
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        for i, mt in enumerate(mts):
            if not test_xs_ref[i].any():
                continue
            import matplotlib.pyplot as plt
            fig, ax1 = plt.subplots()
            lns1 = ax1.loglog(test_energy, test_xs_ref[i], 'g', label="ACE xs")
            lns2 = ax1.loglog(test_energy, best_test_xs[i], 'b', label="VF xs")
            ax2 = ax1.twinx()
            lns3 = ax2.loglog(test_energy, best_relerr[i], 'r',
                              label="Relative error", alpha=0.5)
            lns = lns1 + lns2 + lns3
            labels = [l.get_label() for l in lns]
            ax1.legend(lns, labels, loc='best')
            ax1.set_xlabel('energy (eV)')
            ax1.set_ylabel('cross section (b)', color='b')
            ax1.tick_params('y', colors='b')
            ax2.set_ylabel('relative error', color='r')
            ax2.tick_params('y', colors='r')

            plt.title(f"MT {mt} vector fitted with {mp_poles.size} poles")
            fig.tight_layout()
            fig_file = os.path.join(path_out, "{:.0f}-{:.0f}_MT{}.png".format(
                                    energy[0], energy[-1], mt))
            plt.savefig(fig_file)
            plt.close()
            if log:
                print(f"Saved figure: {fig_file}")

    return (mp_poles, mp_residues, cf)

def _vectfit_moments(energy, ce_xs, mts, rtol=1e-3, atol=1e-5, orders=None,
                n_vf_iter=30, log=False, path_out=None, poly_orders=0, poles_prefit=None,
                widths_prefit=None):
    """Convert point-wise cross section to multipole data via vector fitting.

    Parameters
    ----------
    energy : np.ndarray
        Energy array
    ce_xs : np.ndarray
        Point-wise cross sections to be fitted, with shape (number of reactions,
        number of energy points)
    mts : Iterable of int
        Reaction list
    rtol : float, optional
        Relative error tolerance
    atol : float, optional
        Absolute error tolerance
    orders : Iterable of int, optional
        A list of orders (number of poles) to be searched
    n_vf_iter : int, optional
        Number of maximum VF iterations
    log : bool or int, optional
        Whether to print running logs (use int for verbosity control)
    path_out : str, optional
        Path to save the figures to show discrepancies between the original and
        fitted cross sections for different reactions

    Returns
    -------
    tuple
        (poles, residues)

    """
    
    # import vectfit package: https://github.com/liangjg/vectfit
    import vectfit as vf

    ne = energy.size
    nmt = len(mts)
    if ce_xs.shape != (nmt, ne):
        raise ValueError('Inconsistent cross section data.')

    # construct test data: interpolate xs with finer grids
    n_finer = 10
    ne_test = (ne - 1)*n_finer + 1
    test_energy = np.interp(np.arange(ne_test),
                            np.arange(ne_test, step=n_finer), energy)
    test_energy[[0, -1]] = energy[[0, -1]]  # avoid numerical issue
    test_xs_ref = np.zeros((nmt, ne_test))
    for i in range(nmt):
        test_xs_ref[i] = np.interp(test_energy, energy, ce_xs[i])

    if log:
        print(f"  energy: {energy[0]:.3e} to {energy[-1]:.3e} eV ({ne} points)")
        print(f"  error tolerance: rtol={rtol}, atol={atol}")

    # transform xs (sigma) and energy (E) to f (sigma*E) and s (sqrt(E)) to be
    # compatible with the multipole representation
    f = ce_xs * energy
    s = np.sqrt(energy)
    test_s = np.sqrt(test_energy)

    # inverse weighting is used for minimizing the relative deviation instead of
    # absolute deviation in vector fitting
    with np.errstate(divide='ignore'):
        weight = 1.0/np.abs(f)

    # Cap weights to avoid numerical issues near zero crossings of moments.
    for i in range(nmt):
        finite = weight[i][np.isfinite(weight[i])]
        if finite.size == 0:
            weight[i] = 1.0
        else:
            cap = np.percentile(finite, 95)
            weight[i] = np.minimum(weight[i], cap)
            weight[i] = np.where(np.isfinite(weight[i]), weight[i], cap)

    # detect peaks (resonances) and determine VF order search range
    # normalize each moment by its max so higher-order (smaller) moments contribute equally

    if poles_prefit is None: 
        scale = np.array([np.max(np.abs(xs)) if np.max(np.abs(xs)) > 0 else 1.0 for xs in ce_xs])
        peaks, _ = find_peaks(np.sum(ce_xs / scale[:, np.newaxis], axis=0))
        n_peaks = peaks.size
        if orders is not None:
            # make sure orders are even integers
            orders = list(set([int(i/2)*2 for i in orders if i >= 2]))
        else:
            lowest_order = max(2, 2*n_peaks-2)
            highest_order = 2*n_peaks + 2
            orders = list(range(lowest_order, highest_order + 1, 2))
    
    else:
        peaks = poles_prefit
        n_peaks = peaks.size

        lowest_order = max(2, 2*n_peaks)
        highest_order = 2*n_peaks + 10
        orders = list(range(lowest_order, highest_order + 1, 2))

    if log:
        print(f"Found {n_peaks} peaks")
        print(f"Fitting orders from {orders[0]} to {orders[-1]}")

    # perform VF with increasing orders
    found_ideal = False
    n_discarded = 0  # for accelation, number of discarded searches
    best_quality = best_ratio = -np.inf
    for i, order in enumerate(orders):
        if log:
            print(f"Order={order}({i}/{len(orders)})")
        # initial guessed poles
        if poles_prefit is None: 
            poles_r = np.linspace(s[0], s[-1], order//2)
        
        else:
            poles_r = np.sqrt(poles_prefit)
            widths_r = widths_prefit.copy() if widths_prefit is not None else None
            diff = order - n_peaks * 2

            if diff < 0:
                poles_r = poles_r[:diff//2]
                if widths_r is not None:
                    widths_r = widths_r[:diff//2]

            if diff > 0:
                extra = np.linspace(poles_r[-1], s[-1], diff//2 + 1)[1:]
                poles_r = np.concatenate([poles_r, extra])
                if widths_r is not None:
                    widths_r = np.concatenate([widths_r, extra * extra * 0.04])  # 1% imaginary fallback

            if widths_r is not None:
                # p = sqrt(E_r) + i * Gamma / (4 * sqrt(E_r))
                poles_i = widths_r / (4.0 * poles_r)
            else:
                poles_i = poles_r * 0.01

        poles = poles_r + 1j * poles_i
        poles = np.sort(np.append(poles, np.conj(poles)))

        found_better = False
        # fitting iteration
        for i_vf in range(n_vf_iter):
            if log >= DETAILED_LOGGING:
                print(f"VF iteration {i_vf + 1}/{n_vf_iter}")

            # call vf
            poles, residues, cf, f_fit, rms = vf.vectfit(f, s, poles, weight, n_polys=poly_orders)

            # convert real pole to conjugate pairs
            n_real_poles = 0
            new_poles = []
            for p in poles:
                p_r, p_i = np.real(p), np.imag(p)
                if (s[0] <= p_r <= s[-1]) and p_i == 0.:
                    new_poles += [p_r+p_r*0.01j, p_r-p_r*0.01j]
                    n_real_poles += 1
                else:
                    new_poles += [p]
            new_poles = np.array(new_poles)
            # re-calculate residues if poles changed
            if n_real_poles > 0:
                if log >= DETAILED_LOGGING:
                    print(f"  # real poles: {n_real_poles}")
                new_poles, residues, cf, f_fit, rms = \
                      vf.vectfit(f, s, new_poles, weight, skip_pole=True, n_polys=poly_orders)

            # assess the result on test grid
            test_xs = vf.evaluate(test_s, new_poles, residues, cf) / test_energy
            abserr = np.abs(test_xs - test_xs_ref)

            with np.errstate(invalid='ignore', divide='ignore'):
                relerr = abserr / np.abs(test_xs_ref)
                if np.any(np.isnan(abserr)):
                    maxre, ratio, ratio2 = np.inf, -np.inf, -np.inf
                elif np.all(abserr <= atol):
                    maxre, ratio, ratio2 = 0., 1., 1.
                else:
                    maxre = np.max(relerr[abserr > atol])
                    ratio = np.sum((relerr < rtol) | (abserr < atol)) / relerr.size
                    ratio2 = np.sum((relerr < 10*rtol) | (abserr < atol)) / relerr.size

            # define a metric for choosing the best fitting results
            # basically, it is preferred to have more points within accuracy
            # tolerance, smaller maximum deviation and fewer poles
            #TODO: improve the metric with clearer basis
            quality = ratio + ratio2 - min(0.1*maxre, 1) - 0.001*new_poles.size

            #if np.any(test_xs < -atol): # comment out for now since higher order moments can be negative
            #    quality = -np.inf

            if log >= DETAILED_LOGGING:
                print(f"  # poles: {new_poles.size}")
                print(f"  Max relative error: {maxre * 100:.3f}%")
                print(f"  Satisfaction: {ratio * 100:.1f}%, {ratio2 * 100:.1f}%")
                print(f"  Quality: {quality:.2f}")

            if quality > best_quality:
                if log >= DETAILED_LOGGING:
                    print("  Best so far!")
                found_better = True
                best_quality, best_ratio = quality, ratio
                best_poles, best_residues, best_cf = new_poles, residues, cf
                best_test_xs, best_relerr = test_xs, relerr
                if best_ratio >= 1.0:
                    if log:
                        print("Found ideal results. Stop!")
                    found_ideal = True
                    break
            else:
                if log >= DETAILED_LOGGING:
                    print("  Discarded!")

        if found_ideal:
            break

        # acceleration
        if found_better:
            n_discarded = 0
        else:
            if order > max(2*n_peaks, 50) and best_ratio > 0.7:
                n_discarded += 1
                if n_discarded >= 10 or (n_discarded >= 5 and best_ratio > 0.9):
                    if log >= DETAILED_LOGGING:
                        print("Couldn't get better results. Stop!")
                    break

    # merge conjugate poles
    real_idx = []
    conj_idx = []
    found_conj = False
    for i, p in enumerate(best_poles):
        if found_conj:
            found_conj = False
            continue
        if np.imag(p) == 0.:
            real_idx.append(i)
        else:
            if i < best_poles.size and np.conj(p) == best_poles[i + 1]:
                found_conj = True
                conj_idx.append(i)
            else:
                raise RuntimeError("Complex poles are not conjugate!")
    if log:
        print("Found {} real poles and {} conjugate complex pairs.".format(
               len(real_idx), len(conj_idx)))
    mp_poles = best_poles[real_idx + conj_idx]
    mp_residues = np.concatenate((best_residues[:, real_idx],
                                  best_residues[:, conj_idx]*2), axis=1)/1j
    if log:
        print(f"Final number of poles: {mp_poles.size}")

    if path_out:
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        for i, mt in enumerate(mts):
            if not test_xs_ref[i].any():
                continue
            import matplotlib.pyplot as plt
            fig, ax1 = plt.subplots()
            lns1 = ax1.loglog(test_energy, test_xs_ref[i], 'g', label="ACE xs")
            lns2 = ax1.loglog(test_energy, best_test_xs[i], 'b', label="VF xs")
            ax2 = ax1.twinx()
            lns3 = ax2.loglog(test_energy, best_relerr[i], 'r',
                              label="Relative error", alpha=0.5)
            lns = lns1 + lns2 + lns3
            labels = [l.get_label() for l in lns]
            ax1.legend(lns, labels, loc='best')
            ax1.set_xlabel('energy (eV)')
            ax1.set_ylabel('cross section (b)', color='b')
            ax1.tick_params('y', colors='b')
            ax2.set_ylabel('relative error', color='r')
            ax2.tick_params('y', colors='r')

            plt.title(f"MT {mt} vector fitted with {mp_poles.size} poles")
            fig.tight_layout()
            fig_file = os.path.join(path_out, "{:.0f}-{:.0f}_MT{}.png".format(
                                    energy[0], energy[-1], mt))
            plt.savefig(fig_file)
            plt.close()
            if log:
                print(f"Saved figure: {fig_file}")

    return mp_poles, mp_residues, best_cf

def vectfit_nuclide(endf_file, njoy_error=5e-4, vf_pieces=None,
                    log=False, path_out=None, mp_filename=None, **kwargs):
    r"""Generate multipole data for a nuclide from ENDF.

    Parameters
    ----------
    endf_file : str
        Path to ENDF evaluation
    njoy_error : float, optional
        Fractional error tolerance for processing point-wise data with NJOY
    vf_pieces : integer, optional
        Number of equal-in-momentum spaced energy pieces for data fitting
    log : bool or int, optional
        Whether to print running logs (use int for verbosity control)
    path_out : str, optional
        Path to write out mutipole data file and vector fitting figures
    mp_filename : str, optional
        File name to write out multipole data
    **kwargs
        Keyword arguments passed to :func:`openmc.data.multipole._vectfit_xs`

    Returns
    -------
    mp_data
        Dictionary containing necessary multipole data of the nuclide

    """

    # ======================================================================
    # PREPARE POINT-WISE XS

    # make 0K ACE data using njoy
    if log:
        print(f"Running NJOY to get 0K point-wise data (error={njoy_error})...")

    nuc_ce = IncidentNeutron.from_njoy(endf_file, temperatures=[0.0],
             error=njoy_error, broadr=False, heatr=False, purr=False)

    if log:
        print("Parsing cross sections within resolved resonance range...")

    # Determine upper energy: the lower of RRR upper bound and first threshold
    endf_res = IncidentNeutron.from_endf(endf_file).resonances
    if hasattr(endf_res, 'resolved') and \
       hasattr(endf_res.resolved, 'energy_max') and \
       type(endf_res.resolved) is not ResonanceRange:
        E_max = endf_res.resolved.energy_max
    elif hasattr(endf_res, 'unresolved') and \
         hasattr(endf_res.unresolved, 'energy_min'):
        E_max = endf_res.unresolved.energy_min
    else:
        E_max = nuc_ce.energy['0K'][-1]
    E_max_idx = np.searchsorted(nuc_ce.energy['0K'], E_max, side='right') - 1
    for mt in nuc_ce.reactions:
        if hasattr(nuc_ce.reactions[mt].xs['0K'], '_threshold_idx'):
            threshold_idx = nuc_ce.reactions[mt].xs['0K']._threshold_idx
            if 0 < threshold_idx < E_max_idx:
                E_max_idx = threshold_idx

    # parse energy and cross sections
    energy = nuc_ce.energy['0K'][:E_max_idx + 1]
    E_min, E_max = energy[0], energy[-1]
    n_points = energy.size
    total_xs = nuc_ce[1].xs['0K'](energy)
    elastic_xs = nuc_ce[2].xs['0K'](energy)

    try:
        absorption_xs = nuc_ce[27].xs['0K'](energy)
    except KeyError:
        absorption_xs = np.zeros_like(total_xs)

    fissionable = False
    try:
        fission_xs = nuc_ce[18].xs['0K'](energy)
        fissionable = True
    except KeyError:
        pass

    # make vectors
    if fissionable:
        ce_xs = np.vstack((elastic_xs, absorption_xs, fission_xs))
        mts = [2, 27, 18]
    else:
        ce_xs = np.vstack((elastic_xs, absorption_xs))
        mts = [2, 27]

    if log:
        print(f"  MTs: {mts}")
        print(f"  Energy range: {E_min:.3e} to {E_max:.3e} eV ({n_points} points)")

    # ======================================================================
    # PERFORM VECTOR FITTING

    if vf_pieces is None:
        # divide into pieces for complex nuclides
        peaks, _ = find_peaks(total_xs)
        n_peaks = peaks.size
        if n_peaks > 200 or n_points > 30000 or n_peaks * n_points > 100*10000:
            vf_pieces = max(5, n_peaks // 50,  n_points // 2000)
        else:
            vf_pieces = 1
    piece_width = (sqrt(E_max) - sqrt(E_min)) / vf_pieces

    alpha = nuc_ce.atomic_weight_ratio/(K_BOLTZMANN*TEMPERATURE_LIMIT)

    poles, residues = [], []
    # VF piece by piece
    for i_piece in range(vf_pieces):
        if log:
            print(f"Vector fitting piece {i_piece + 1}/{vf_pieces}...")
        # start E of this piece
        e_bound = (sqrt(E_min) + piece_width*(i_piece-0.5))**2
        if i_piece == 0 or sqrt(alpha*e_bound) < 4.0:
            e_start = E_min
            e_start_idx = 0
        else:
            e_start = max(E_min, (sqrt(alpha*e_bound) - 4.0)**2/alpha)
            e_start_idx = np.searchsorted(energy, e_start, side='right') - 1
        # end E of this piece
        e_bound = (sqrt(E_min) + piece_width*(i_piece + 1))**2
        e_end = min(E_max, (sqrt(alpha*e_bound) + 4.0)**2/alpha)
        e_end_idx = np.searchsorted(energy, e_end, side='left') + 1
        e_idx = range(e_start_idx, min(e_end_idx + 1, n_points))

        p, r, _ = _vectfit_xs(energy[e_idx], ce_xs[:, e_idx], mts, log=log,
                              path_out=path_out, **kwargs)

        poles.append(p)
        residues.append(r)

    # collect multipole data into a dictionary
    mp_data = {"name": nuc_ce.name,
               "AWR": nuc_ce.atomic_weight_ratio,
               "E_min": E_min,
               "E_max": E_max,
               "poles": poles,
               "residues": residues}

    # dump multipole data to file
    if path_out:
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        if not mp_filename:
            mp_filename = f"{nuc_ce.name}_mp.pickle"
        mp_filename = os.path.join(path_out, mp_filename)
        with open(mp_filename, 'wb') as f:
            pickle.dump(mp_data, f)
        if log:
            print(f"Dumped multipole data to file: {mp_filename}")

    return mp_data


def _load_moments(moments_file):
    """Load scattering moments data from a columnar text file.

    Parameters
    ----------
    moments_file : str
        Path to moments data file. First column is energy; remaining columns
        are Legendre moments sig0, sig1, sig2, ... First line is skipped as
        a header.

    Returns
    -------
    energies : np.ndarray
        Energy array, shape (n_energy_points,).
    moments : np.ndarray
        Moments array, shape (n_moments, n_energy_points).

    """
    data = np.loadtxt(moments_file, skiprows=1)
    energies = data[:, 0]
    moments = data[:, 1:].T  # shape (n_moments, n_energy_points)
    return energies, moments


def vectfit_moments_nuclide(moments_file, awr, E_min=None, E_max=None,
                             absorption_file=None,
                             vf_pieces=None, log=False, path_out=None,
                             mp_filename=None, **kwargs):
    """Generate multipole data for scattering moments (and optionally absorption).

    Fits all Legendre moments (sigma_0, sigma_1, ...) simultaneously using
    shared poles via vector fitting. An optional absorption cross section file
    can be supplied; its values are interpolated onto the moments energy grid
    and appended as an extra channel so that all quantities share the same pole
    set.  The file may contain any number of moment columns beyond the mandatory
    energy and sigma_0 columns. The resulting mp_data can be passed to
    :func:`WindowedMultipole.from_multipole` to build a windowed multipole
    library. Use ``poly_orders=0`` (the default) since :func:`_windowing` fits
    its own per-window polynomial.

    Parameters
    ----------
    moments_file : str
        Path to moments data file. First column is energy; remaining columns
        are Legendre moments sigma_0, sigma_1, ... (any number supported).
    awr : float
        Atomic weight ratio of the target nuclide (used for Doppler-broadened
        energy piece boundaries and windowing).
    E_min : float, optional
        Lower energy bound in eV. Defaults to first energy point in file.
    E_max : float, optional
        Upper energy bound in eV. Defaults to last energy point in file.
    absorption_file : str, optional
        Path to a two-column (energy, sigma_abs) text file containing the
        absorption cross section. The values are linearly interpolated onto
        the moments energy grid and appended as the last channel so that
        moments and absorption share a common pole set.
    vf_pieces : int, optional
        Number of equal-in-momentum energy pieces for fitting. Auto-selected
        if not provided.
    log : bool or int, optional
        Whether to print running logs.
    path_out : str, optional
        Directory path to save figures and the pickle output file.
    mp_filename : str, optional
        Filename for the pickle output. Defaults to ``moments_mp.pickle``.
    **kwargs
        Keyword arguments passed to :func:`_vectfit_moments`.

    Returns
    -------
    dict
        Multipole data dictionary with keys ``name``, ``AWR``, ``E_min``,
        ``E_max``, ``poles``, and ``residues`` — same format as
        :func:`vectfit_nuclide`.

    """
    # Load moments data — shape (n_moments, n_energy_points)
    energies, ce_xs = _load_moments(moments_file)

    # Optionally append absorption as an extra channel
    if absorption_file is not None:
        abs_data = np.loadtxt(absorption_file, skiprows=1)
        abs_energies = abs_data[:, 0]
        abs_xs = abs_data[:, 1]
        abs_interp = np.interp(energies, abs_energies, abs_xs)
        ce_xs = np.vstack([ce_xs, abs_interp])

    n_channels = ce_xs.shape[0]
    mts = list(range(n_channels))

    # Filter to requested energy range
    mask = np.ones(energies.size, dtype=bool)
    if E_min is not None:
        mask &= energies >= E_min
    if E_max is not None:
        mask &= energies <= E_max
    energies = energies[mask]
    ce_xs = ce_xs[:, mask]

    n_points = energies.size
    E_min_data = energies[0]
    E_max_data = energies[-1]

    if log:
        n_moments = n_channels - (1 if absorption_file is not None else 0)
        print(f"Loaded {n_moments} moment(s)"
              + (f" + absorption" if absorption_file is not None else "")
              + f": {n_points} points from {E_min_data:.3e} to {E_max_data:.3e} eV")

    # Determine number of pieces (peak detection on sigma_0)
    if vf_pieces is None:
        peaks, _ = find_peaks(ce_xs[0])
        n_peaks = peaks.size
        if n_peaks > 200 or n_points > 30000 or n_peaks * n_points > 100*10000:
            vf_pieces = max(5, n_peaks // 50, n_points // 2000)
        else:
            vf_pieces = 1

    piece_width = (sqrt(E_max_data) - sqrt(E_min_data)) / vf_pieces
    alpha = awr / (K_BOLTZMANN * TEMPERATURE_LIMIT)

    poles, residues = [], []
    for i_piece in range(vf_pieces):
        if log:
            print(f"Vector fitting piece {i_piece + 1}/{vf_pieces}...")

        # start E of this piece (overlap for Doppler broadening, same as vectfit_nuclide)
        e_bound = (sqrt(E_min_data) + piece_width * (i_piece - 0.5))**2
        if i_piece == 0 or sqrt(alpha * e_bound) < 4.0:
            e_start = E_min_data
            e_start_idx = 0
        else:
            e_start = max(E_min_data, (sqrt(alpha * e_bound) - 4.0)**2 / alpha)
            e_start_idx = np.searchsorted(energies, e_start, side='right') - 1

        # end E of this piece
        e_bound = (sqrt(E_min_data) + piece_width * (i_piece + 1))**2
        e_end = min(E_max_data, (sqrt(alpha * e_bound) + 4.0)**2 / alpha)
        e_end_idx = np.searchsorted(energies, e_end, side='left') + 1
        e_idx = range(e_start_idx, min(e_end_idx + 1, n_points))

        p, r, _ = _vectfit_moments(energies[e_idx], ce_xs[:, e_idx], mts,
                                    log=log, path_out=path_out, **kwargs)
        poles.append(p)
        residues.append(r)

    mp_data = {
        "name": "moments",
        "AWR": awr,
        "E_min": E_min_data,
        "E_max": E_max_data,
        "poles": poles,
        "residues": residues,
    }

    if path_out:
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        if not mp_filename:
            mp_filename = "moments_mp.pickle"
        mp_filename = os.path.join(path_out, mp_filename)
        with open(mp_filename, 'wb') as f:
            pickle.dump(mp_data, f)
        if log:
            print(f"Dumped moments multipole data to file: {mp_filename}")

    return mp_data


def _fit_pseudopoles(energy_sqrt, energy, residual, xs_ref, rtol, atol,
                     n_channels):
    """Fit residual cross section with pseudopoles via vector fitting.

    Returns poles and residues in multipole convention (residues / 1j).
    Returns empty arrays if the residual is below tolerance or fitting fails.
    """
    import vectfit as vf

    max_ref = np.max(np.abs(xs_ref))
    max_res = np.max(np.abs(residual))
    if max_ref == 0 or max_res / max_ref <= atol:
        return np.array([]), np.zeros((n_channels, 0))

    s = energy_sqrt
    f_residual = residual * energy

    with np.errstate(divide='ignore'):
        weight = 1.0 / np.abs(f_residual)
    for ic in range(n_channels):
        finite = weight[ic][np.isfinite(weight[ic])]
        if finite.size == 0:
            weight[ic] = 1.0
        else:
            cap = np.percentile(finite, 95)
            weight[ic] = np.minimum(weight[ic], cap)
            weight[ic] = np.where(np.isfinite(weight[ic]), weight[ic], cap)

    best_poles = np.array([])
    best_residues = np.zeros((n_channels, 0))
    best_err = np.inf

    for order in range(2, 12, 2):
        try:
            poles_r = np.linspace(s[0], s[-1], order // 2)
            init_poles = poles_r + poles_r * 0.01j
            init_poles = np.sort(np.append(init_poles, np.conj(init_poles)))
            pp, pr, _, _, _ = vf.vectfit(f_residual, s, init_poles, weight)
        except (RuntimeError, np.linalg.LinAlgError):
            continue

        xs_pseudo = vf.evaluate(s, pp, pr) / energy
        abserr = np.abs(residual - xs_pseudo)
        with np.errstate(invalid='ignore', divide='ignore'):
            relerr = abserr / np.abs(xs_ref)
        re = relerr[abserr > atol]
        max_re = re.max() if re.size > 0 else 0.0

        if max_re < best_err:
            best_poles, best_residues, best_err = pp, pr, max_re

        if re.size == 0 or max_re <= rtol:
            break

    if best_poles.size == 0:
        return np.array([]), np.zeros((n_channels, 0))

    # merge conjugate pairs into multipole convention (residues / 1j)
    real_idx = []
    conj_idx = []
    found_conj = False
    for i, p in enumerate(best_poles):
        if found_conj:
            found_conj = False
            continue
        if np.imag(p) == 0.:
            real_idx.append(i)
        else:
            if (i + 1 < best_poles.size and
                    np.isclose(np.conj(p), best_poles[i + 1])):
                found_conj = True
                conj_idx.append(i)
            else:
                real_idx.append(i)

    mp_poles = best_poles[real_idx + conj_idx]
    mp_residues = np.concatenate(
        (best_residues[:, real_idx],
         best_residues[:, conj_idx] * 2), axis=1) / 1j

    return mp_poles, mp_residues


def _windowing(mp_data, rtol=1e-3, atol=1e-5, n_win=None, spacing=None,
               log=False):
    """Generate windowed multipole library from multipole data using pseudopoles.

    Parameters
    ----------
    mp_data : dict
        Multipole data
    rtol : float, optional
        Maximum relative error tolerance
    atol : float, optional
        Minimum absolute error tolerance
    n_win : int, optional
        Number of equal-in-momentum spaced energy windows
    spacing : float, optional
        Inner window spacing (sqrt energy space)
    log : bool or int, optional
        Whether to print running logs (use int for verbosity control)

    Returns
    -------
    openmc.data.WindowedMultipole
        Resonant cross sections represented in the windowed multipole format.

    """

    # import vectfit package: https://github.com/liangjg/vectfit
    import vectfit as vf

    # unpack multipole data
    name = mp_data["name"]
    awr = mp_data["AWR"]
    E_min = mp_data["E_min"]
    E_max = mp_data["E_max"]
    mp_poles = mp_data["poles"]
    mp_residues = mp_data["residues"]

    n_pieces = len(mp_poles)
    piece_width = (sqrt(E_max) - sqrt(E_min)) / n_pieces
    alpha = awr / (K_BOLTZMANN * TEMPERATURE_LIMIT)
    n_channels = mp_residues[0].shape[0]

    # determine window size
    if n_win is None:
        if spacing is not None:
            n_win = int((sqrt(E_max) - sqrt(E_min)) / spacing)
            E_max = (sqrt(E_min) + n_win * spacing)**2
        else:
            n_win = 1000
    spacing = (sqrt(E_max) - sqrt(E_min)) / n_win
    if spacing > piece_width:
        raise ValueError('Window spacing cannot be larger than piece spacing.')

    if log:
        print("Windowing:")
        print(f"  config: # windows={n_win}, spacing={spacing}")
        print(f"  error tolerance: rtol={rtol}, atol={atol}")

    # sort poles (and residues) by the real component of the pole
    for ip in range(n_pieces):
        indices = mp_poles[ip].argsort()
        mp_poles[ip] = mp_poles[ip][indices]
        mp_residues[ip] = mp_residues[ip][:, indices]

    # track which poles are referenced by at least one window
    poles_unused = [np.ones_like(p, dtype=int) for p in mp_poles]

    win_data = []       # (i_piece, lp, rp, ps_start, ps_end) per window
    pseudo_rows = []    # list of (1+n_channels,) complex arrays, one per pseudopole

    for iw in range(n_win):
        if log >= DETAILED_LOGGING:
            print(f"Processing window {iw + 1}/{n_win}...")

        # inner window boundaries
        inbegin = sqrt(E_min) + spacing * iw
        inend = inbegin + spacing
        incenter = (inbegin + inend) / 2.0
        # extend window energy range for Doppler broadening
        if iw == 0 or sqrt(alpha) * inbegin < 4.0:
            e_start = inbegin**2
        else:
            e_start = max(E_min, (sqrt(alpha) * inbegin - 4.0)**2 / alpha)
        e_end = min(E_max, (sqrt(alpha) * inend + 4.0)**2 / alpha)

        # locate piece and relevant poles
        i_piece = min(n_pieces - 1, int((inbegin - sqrt(E_min)) / piece_width + 0.5))
        poles, residues = mp_poles[i_piece], mp_residues[i_piece]
        n_poles = poles.size

        # generate energy points for fitting: equally spaced in momentum
        n_points = min(max(100, int((e_end - e_start) * 4)), 10000)
        energy_sqrt = np.linspace(np.sqrt(e_start), np.sqrt(e_end), n_points)
        energy = energy_sqrt**2

        # reference xs from multipole form
        xs_ref = vf.evaluate(energy_sqrt, poles, residues * 1j) / energy

        # start from 0 poles, initialize pointers to the center nearest pole
        center_pole_ind = np.argmin(np.fabs(poles.real - incenter))
        lp = rp = center_pole_ind
        while True:
            if log >= DETAILED_LOGGING:
                print(f"Trying poles {lp} to {rp}")

            if rp > lp:
                xs_wp = vf.evaluate(energy_sqrt, poles[lp:rp],
                                    residues[:, lp:rp] * 1j) / energy
            else:
                xs_wp = np.zeros_like(xs_ref)

            # assess residual after real poles
            residual = xs_ref - xs_wp
            abserr = np.abs(residual)
            with np.errstate(invalid='ignore', divide='ignore'):
                relerr = abserr / np.abs(xs_ref)
            if not np.any(np.isnan(abserr)):
                re = relerr[abserr > atol]
                if re.size == 0 or np.all(re <= rtol) or \
                   (re.max() <= 2 * rtol and (re > rtol).sum() <= 0.01 * relerr.size) or \
                   (iw == 0 and np.all(relerr.mean(axis=1) <= rtol)):
                    if log >= DETAILED_LOGGING:
                        print("Accuracy satisfied.")
                    break

            if rp >= n_poles and lp <= 0:
                break  # no more poles available
            elif rp >= n_poles:
                lp -= 1
            elif lp <= 0 or poles[rp] - incenter <= incenter - poles[lp - 1]:
                rp += 1
            else:
                lp -= 1

        # fit residual with pseudopoles
        ps_start = len(pseudo_rows)
        residual = xs_ref - xs_wp
        pseudo_poles_win, pseudo_residues_win = _fit_pseudopoles(
            energy_sqrt, energy, residual, xs_ref, rtol, atol, n_channels)
        if pseudo_poles_win.size > 0:
            if log >= DETAILED_LOGGING:
                print(f"  Added {pseudo_poles_win.size} pseudopole(s)")
            for j in range(pseudo_poles_win.size):
                row = np.concatenate(
                    [[pseudo_poles_win[j]], pseudo_residues_win[:, j]])
                pseudo_rows.append(row)
        ps_end = len(pseudo_rows)

        win_data.append((i_piece, lp, rp, ps_start, ps_end))
        poles_unused[i_piece][lp:rp] = 0

    # flatten and remove unused poles
    data_parts = []
    for ip in range(n_pieces):
        used = (poles_unused[ip] == 0)
        if used.any():
            data_parts.append(
                np.vstack([mp_poles[ip][used], mp_residues[ip][:, used]]).T)
    data = (np.vstack(data_parts) if data_parts
            else np.zeros((0, 1 + n_channels), dtype=complex))

    # build pseudo_data array
    pseudo_data = (np.array(pseudo_rows) if pseudo_rows
                   else np.zeros((0, 1 + n_channels), dtype=complex))

    # build 4-column windows array with 0-based indices
    windows = []
    for iw in range(n_win):
        ip, lp, rp, ps_start, ps_end = win_data[iw]
        n_prev_poles = sum(mp_poles[i].size for i in range(ip))
        n_unused = sum((poles_unused[i] == 1).sum() for i in range(ip)) + \
                   (poles_unused[ip][:lp] == 1).sum()
        lp_global = lp + n_prev_poles - n_unused
        rp_global = rp + n_prev_poles - n_unused
        windows.append([lp_global, rp_global, ps_start, ps_end])

    # construct the WindowedMultipole object
    wmp = WindowedMultipole(name)
    wmp.sqrtAWR = sqrt(awr)
    wmp.E_min = E_min
    wmp.E_max = E_max
    wmp.data = data
    wmp.pseudo_data = pseudo_data
    wmp.windows = np.asarray(windows, dtype=int)

    return wmp


class WindowedMultipole(EqualityMixin):
    """Resonant cross sections represented in the windowed multipole format.

    Parameters
    ----------
    name : str
        Name of the nuclide using the GNDS naming convention

    Attributes
    ----------
    name : str
        Name of the nuclide using the GNDS naming convention
    spacing : float
        The width of each window in sqrt(E)-space.  For example, the frst window
        will end at (sqrt(E_min) + spacing)**2 and the second window at
        (sqrt(E_min) + 2*spacing)**2.
    sqrtAWR : float
        Square root of the atomic weight ratio of the target nuclide.
    E_min : float
        Lowest energy in eV the library is valid for.
    E_max : float
        Highest energy in eV the library is valid for.
    data : np.ndarray
        A 2D array of complex poles and residues.  data[i, 0] gives the energy
        at which pole i is located.  data[i, 1:] gives the residues associated
        with the i-th pole.  There are 3 residues, one each for the scattering,
        absorption, and fission channels.
    windows : np.ndarray
        A 2D array of Integral values.  windows[i, 0] - 1 is the index of the
        first pole in window i. windows[i, 1] - 1 is the index of the last pole
        in window i.
    broaden_poly : np.ndarray
        A 1D array of boolean values indicating whether or not the polynomial
        curvefit in that window should be Doppler broadened.
    curvefit : np.ndarray
        A 3D array of Real curvefit polynomial coefficients.  curvefit[i, 0, :]
        gives coefficients for the scattering cross section in window i.
        curvefit[i, 1, :] gives absorption coefficients and curvefit[i, 2, :]
        gives fission coefficients.  The polynomial terms are increasing powers
        of sqrt(E) starting with 1/E e.g:
        a/E + b/sqrt(E) + c + d sqrt(E) + ...

    """
    def __init__(self, name):
        self.name = name
        self.sqrtAWR = None
        self.E_min = None
        self.E_max = None
        self.data = None
        self.pseudo_data = None
        self.windows = None
        self.broaden_poly = None
        self.curvefit = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        cv.check_type('name', name, str)
        self._name = name

    @property
    def fit_order(self):
        return self.curvefit.shape[1] - 1

    @property
    def fissionable(self):
        return self.data.shape[1] == 4

    @property
    def n_poles(self):
        return self.data.shape[0]

    @property
    def n_windows(self):
        return self.windows.shape[0]

    @property
    def poles_per_window(self):
        return (self.windows[:, 1] - self.windows[:, 0]).mean()

    @property
    def pseudo_per_window(self):
        if (self.pseudo_data is None or self.windows is None
                or self.windows.shape[1] < 4):
            return 0.0
        return (self.windows[:, 3] - self.windows[:, 2]).mean()

    @property
    def spacing(self):
        return (np.sqrt(self.E_max) - np.sqrt(self.E_min)) / self.n_windows

    @property
    def sqrtAWR(self):
        return self._sqrtAWR

    @sqrtAWR.setter
    def sqrtAWR(self, sqrtAWR):
        if sqrtAWR is not None:
            cv.check_type('sqrtAWR', sqrtAWR, Real)
            cv.check_greater_than('sqrtAWR', sqrtAWR, 0.0, equality=False)
        self._sqrtAWR = sqrtAWR

    @property
    def E_min(self):
        return self._E_min

    @E_min.setter
    def E_min(self, E_min):
        if E_min is not None:
            cv.check_type('E_min', E_min, Real)
            cv.check_greater_than('E_min', E_min, 0.0, equality=True)
        self._E_min = E_min

    @property
    def E_max(self):
        return self._E_max

    @E_max.setter
    def E_max(self, E_max):
        if E_max is not None:
            cv.check_type('E_max', E_max, Real)
            cv.check_greater_than('E_max', E_max, 0.0, equality=False)
        self._E_max = E_max

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if data is not None:
            cv.check_type('data', data, np.ndarray)
            if len(data.shape) != 2:
                raise ValueError('Multipole data arrays must be 2D')
            if data.shape[1] < 3:
                raise ValueError(
                     'data.shape[1] must be at least 3: one value for the pole'
                     ' and at least two for the cross-section or moment residues.')
            if not np.issubdtype(data.dtype, np.complexfloating):
                raise TypeError('Multipole data arrays must be complex dtype')
        self._data = data

    @property
    def pseudo_data(self):
        return self._pseudo_data

    @pseudo_data.setter
    def pseudo_data(self, pseudo_data):
        if pseudo_data is not None:
            cv.check_type('pseudo_data', pseudo_data, np.ndarray)
            if len(pseudo_data.shape) != 2:
                raise ValueError('Multipole pseudo_data arrays must be 2D')
            if pseudo_data.shape[1] < 2:
                raise ValueError(
                    'pseudo_data.shape[1] must be at least 2: one value for '
                    'the pole and at least one residue.')
            if not np.issubdtype(pseudo_data.dtype, np.complexfloating):
                raise TypeError('Multipole pseudo_data arrays must be complex dtype')
        self._pseudo_data = pseudo_data

    @property
    def windows(self):
        return self._windows

    @windows.setter
    def windows(self, windows):
        if windows is not None:
            cv.check_type('windows', windows, np.ndarray)
            if len(windows.shape) != 2:
                raise ValueError('Multipole windows arrays must be 2D')
            if windows.shape[1] not in (2, 4):
                raise ValueError('Multipole windows arrays must have 2 or 4 '
                                 'columns (2: poles only; 4: poles + pseudopoles)')
            if not np.issubdtype(windows.dtype, np.integer):
                raise TypeError('Multipole windows arrays must be integer dtype')
        self._windows = windows

    @property
    def broaden_poly(self):
        return self._broaden_poly
    
    @broaden_poly.setter
    def broaden_poly(self, broaden_poly):
        if broaden_poly is not None:
            cv.check_type('broaden_poly', broaden_poly, np.ndarray)
            if len(broaden_poly.shape) != 1:
                raise ValueError('Multipole broaden_poly arrays must be 1D')
            if not np.issubdtype(broaden_poly.dtype, np.bool_):
                raise TypeError('Multipole broaden_poly arrays must be boolean'
                                ' dtype')
        self._broaden_poly = broaden_poly

    @property
    def curvefit(self):
        return self._curvefit

    @curvefit.setter
    def curvefit(self, curvefit):
        if curvefit is not None:
            cv.check_type('curvefit', curvefit, np.ndarray)
            if len(curvefit.shape) != 3:
                raise ValueError('Multipole curvefit arrays must be 3D')
            if curvefit.shape[2] < 2:
                raise ValueError('The third dimension of multipole curvefit'
                                 ' arrays must have a length of at least 2')
            if not np.issubdtype(curvefit.dtype, np.floating):
                raise TypeError('Multipole curvefit arrays must be float dtype')
        self._curvefit = curvefit

    @classmethod
    def from_hdf5(cls, group_or_filename):
        """Construct a WindowedMultipole object from an HDF5 group or file.

        Parameters
        ----------
        group_or_filename : h5py.Group or str
            HDF5 group containing multipole data. If given as a string, it is
            assumed to be the filename for the HDF5 file, and the first group is
            used to read from.

        Returns
        -------
        openmc.data.WindowedMultipole
            Resonant cross sections represented in the windowed multipole
            format.

        """

        if isinstance(group_or_filename, h5py.Group):
            group = group_or_filename
            need_to_close = False
        else:
            h5file = h5py.File(str(group_or_filename), 'r')
            need_to_close = True

            # Make sure version matches
            if 'version' in h5file.attrs:
                major, minor = h5file.attrs['version']
                if major != WMP_VERSION_MAJOR:
                    raise DataError(
                        'WMP data format uses version {}. {} whereas your '
                        'installation of the OpenMC Python API expects version '
                        '{}.x.'.format(major, minor, WMP_VERSION_MAJOR))
            #else:
            #    raise DataError(
            #        'WMP data does not indicate a version. Your installation of '
            #        'the OpenMC Python API expects version {}.x data.'
            #        .format(WMP_VERSION_MAJOR))

            group = list(h5file.values())[0]

        name = group.name[1:]
        out = cls(name)

        # Read scalars.
        out.sqrtAWR = group['sqrtAWR'][()]
        out.E_min = group['E_min'][()]
        out.E_max = group['E_max'][()]

        # Read arrays.
        out.data = group['data'][()]

        if 'pseudo_data' in group:
            out.pseudo_data = group['pseudo_data'][()]

        out.windows = group['windows'][()]

        if 'broaden_poly' in group:
            out.broaden_poly = group['broaden_poly'][...].astype(bool)

        if 'curvefit' in group:
            out.curvefit = group['curvefit'][()]

        # If HDF5 file was opened here, make sure it gets closed
        if need_to_close:
            h5file.close()

        return out

    @classmethod
    def from_endf(cls, endf_file, log=False, vf_options=None, wmp_options=None):
        """Generate windowed multipole neutron data from an ENDF evaluation.

        .. versionadded:: 0.12.1

        Parameters
        ----------
        endf_file : str
            Path to ENDF evaluation
        log : bool or int, optional
            Whether to print running logs (use int for verbosity control)
        vf_options : dict, optional
            Dictionary of keyword arguments, e.g. {'njoy_error': 0.001},
            passed to :func:`openmc.data.multipole.vectfit_nuclide`
        wmp_options : dict, optional
            Dictionary of keyword arguments, e.g. {'search': True, 'rtol': 0.01},
            passed to :func:`openmc.data.WindowedMultipole.from_multipole`

        Returns
        -------
        openmc.data.WindowedMultipole
            Resonant cross sections represented in the windowed multipole
            format.

        """

        if vf_options is None:
            vf_options = {}

        if wmp_options is None:
            wmp_options = {}

        if log:
            vf_options.update(log=log)
            wmp_options.update(log=log)

        # generate multipole data from EDNF
        mp_data = vectfit_nuclide(endf_file, **vf_options)

        # windowing
        return cls.from_multipole(mp_data, **wmp_options)

    @classmethod
    def from_multipole(cls, mp_data, search=None, log=False, **kwargs):
        """Generate windowed multipole neutron data from multipole data.

        Parameters
        ----------
        mp_data : dictionary or str
            Dictionary or Path to the multipole data stored in a pickle file
        search : bool, optional
            Whether to search for optimal window size and curvefit order.
            Defaults to True if no windowing parameters are specified.
        log : bool or int, optional
            Whether to print running logs (use int for verbosity control)
        **kwargs
            Keyword arguments passed to :func:`openmc.data.multipole._windowing`

        Returns
        -------
        openmc.data.WindowedMultipole
            Resonant cross sections represented in the windowed multipole
            format.

        """

        if isinstance(mp_data, str):
            # load multipole data from file
            with open(mp_data, 'rb') as f:
                mp_data = pickle.load(f)

        if search is None:
            if 'n_win' in kwargs or 'spacing' in kwargs:
                search = False
            else:
                search = True

        # windowing with specific options
        if not search:
            kwargs.pop('n_cf', None)  # n_cf no longer used
            return _windowing(mp_data, log=log, **kwargs)

        # search optimal WMP over a range of window sizes
        if log:
            print("Start searching ...")
        n_poles = sum([p.size for p in mp_data["poles"]])
        n_win_min = max(5, n_poles // 20)
        n_win_max = 2000 if n_poles < 2000 else 8000
        best_wmp = best_metric = None
        for n_w in np.unique(np.linspace(n_win_min, n_win_max, 20, dtype=int)):
            if log:
                print(f"Testing N_win={n_w}")

            kwargs.update(n_win=n_w)

            try:
                wmp = _windowing(mp_data, log=log, **kwargs)
            except Exception as e:
                if log:
                    print('Failed: ' + str(e))
                continue

            # metric: fewer real poles per window and fewer pseudopoles is better
            metric = -(wmp.poles_per_window * 10. + wmp.pseudo_per_window * 5. +
                       wmp.n_windows * 0.01)
            if best_wmp is None or metric > best_metric:
                if log:
                    print("Best library so far.")
                best_wmp = deepcopy(wmp)
                best_metric = metric

        # return the best wmp library
        if log:
            print("Final library: {} poles, {} windows, {:.2g} poles/window, "
                  "{:.2g} pseudo/window".format(
                  best_wmp.n_poles, best_wmp.n_windows,
                  best_wmp.poles_per_window, best_wmp.pseudo_per_window))

        return best_wmp

    def _evaluate(self, E, T):
        """Compute scattering, absorption, and fission cross sections.

        Parameters
        ----------
        E : Real
            Energy of the incident neutron in eV.
        T : Real
            Temperature of the target in K.

        Returns
        -------
        3-tuple of Real
            Scattering, absorption, and fission microscopic cross sections
            at the given energy and temperature.

        """

        n_channels = self.data.shape[1] - 1
        if E < self.E_min: return tuple([0.0] * n_channels)
        if E > self.E_max: return tuple([0.0] * n_channels)

        # ======================================================================
        # Bookkeeping

        sqrtkT = sqrt(K_BOLTZMANN * T)
        sqrtE = sqrt(E)
        invE = 1.0 / E

        # windows use 0-based indices; endw is exclusive
        i_window = min(self.n_windows - 1,
                       int(np.floor((sqrtE - sqrt(self.E_min)) / self.spacing)))
        startw = self.windows[i_window, 0]
        endw = self.windows[i_window, 1]

        # Initialize the output cross sections (one per channel).
        sigs = [0.0] * n_channels

        # ======================================================================
        # Add the contribution from the curvefit polynomial (backward compat).

        if self.curvefit is not None:
            if sqrtkT != 0 and self.broaden_poly is not None and self.broaden_poly[i_window]:
                dopp = self.sqrtAWR / sqrtkT
                broadened_polynomials = _broaden_wmp_polynomials(E, dopp,
                                                                 self.fit_order + 1)
                for i_poly in range(self.fit_order + 1):
                    for i_chan in range(n_channels):
                        sigs[i_chan] += (self.curvefit[i_window, i_poly, i_chan]
                                        * broadened_polynomials[i_poly])
            else:
                temp = invE
                for i_poly in range(self.fit_order + 1):
                    for i_chan in range(n_channels):
                        sigs[i_chan] += self.curvefit[i_window, i_poly, i_chan] * temp
                    temp *= sqrtE

        # ======================================================================
        # Add the contribution from the real poles in this window.

        if sqrtkT == 0.0:
            # If at 0K, use asymptotic form.
            for i_pole in range(startw, endw):
                psi_chi = -1j / (self.data[i_pole, _MP_EA] - sqrtE)
                c_temp = psi_chi / E
                for i_chan in range(n_channels):
                    sigs[i_chan] += (self.data[i_pole, i_chan + 1] * c_temp).real

        else:
            # At temperature, use Faddeeva function-based form.
            dopp = self.sqrtAWR / sqrtkT
            for i_pole in range(startw, endw):
                Z = (sqrtE - self.data[i_pole, _MP_EA]) * dopp
                w_val = _faddeeva(Z) * dopp * invE * sqrt(pi)
                for i_chan in range(n_channels):
                    sigs[i_chan] += (self.data[i_pole, i_chan + 1] * w_val).real

        return tuple(sigs)

    def _evaluate_pseudo(self, E, T):
        """Compute Doppler-broadened Legendre moments from pseudopole data.

        Uses the same asymptotic (0K) / Faddeeva-function (T>0) forms as the
        physical poles in :meth:`_evaluate`, applied to each pseudopole
        residue channel (e.g. sigma_0, sigma_1, sigma_2, ... angular
        moments).

        Parameters
        ----------
        E : Real
            Energy of the incident neutron in eV.
        T : Real
            Temperature of the target in K.

        Returns
        -------
        tuple of Real
            Legendre moment values, one per residue channel in pseudo_data.
            Empty if this nuclide has no pseudopole data.

        """

        if self.pseudo_data is None or self.windows.shape[1] < 4:
            return ()

        n_moments = self.pseudo_data.shape[1] - 1
        if E < self.E_min or E > self.E_max:
            return tuple([0.0] * n_moments)

        sqrtkT = sqrt(K_BOLTZMANN * T)
        sqrtE = sqrt(E)
        invE = 1.0 / E

        i_window = min(self.n_windows - 1,
                       int(np.floor((sqrtE - sqrt(self.E_min)) / self.spacing)))
        startpw = self.windows[i_window, 2]
        endpw = self.windows[i_window, 3]

        moments = [0.0] * n_moments

        if sqrtkT == 0.0:
            # If at 0K, use asymptotic form.
            for i_pseudo in range(startpw, endpw):
                psi_chi = -1j / (self.pseudo_data[i_pseudo, 0] - sqrtE)
                c_temp = psi_chi * invE
                for i_mom in range(n_moments):
                    moments[i_mom] += (self.pseudo_data[i_pseudo, i_mom + 1]
                                       * c_temp).real
        else:
            # At temperature, use Faddeeva function-based form.
            dopp = self.sqrtAWR / sqrtkT
            for i_pseudo in range(startpw, endpw):
                Z = (sqrtE - self.pseudo_data[i_pseudo, 0]) * dopp
                w_val = _faddeeva(Z) * dopp * invE * sqrt(pi)
                for i_mom in range(n_moments):
                    moments[i_mom] += (self.pseudo_data[i_pseudo, i_mom + 1]
                                       * w_val).real

        return tuple(moments)

    def __call__(self, E, T):
        """Compute scattering, absorption, and fission cross sections.

        Parameters
        ----------
        E : Real or Iterable of Real
            Energy of the incident neutron in eV.
        T : Real
            Temperature of the target in K.

        Returns
        -------
        3-tuple of Real or 3-tuple of numpy.ndarray
            Scattering, absorption, and fission microscopic cross sections
            at the given energy and temperature.

        """

        fun = np.vectorize(lambda x: self._evaluate(x, T))
        return fun(E)

    def export_to_hdf5(self, path, mode='a', libver='earliest'):
        """Export windowed multipole data to an HDF5 file.

        Parameters
        ----------
        path : str
            Path to write HDF5 file to
        mode : {'r+', 'w', 'x', 'a'}
            Mode that is used to open the HDF5 file. This is the second argument
            to the :class:`h5py.File` constructor.
        libver : {'earliest', 'latest'}
            Compatibility mode for the HDF5 file. 'latest' will produce files
            that are less backwards compatible but have performance benefits.

        """

        # Open file and write version.
        with h5py.File(str(path), mode, libver=libver) as f:
            f.attrs['filetype'] = np.bytes_('data_wmp')
            f.attrs['version'] = np.array(WMP_VERSION)

            g = f.create_group(self.name)

            # Write scalars.
            g.create_dataset('sqrtAWR', data=np.array(self.sqrtAWR))
            g.create_dataset('E_min', data=np.array(self.E_min))
            g.create_dataset('E_max', data=np.array(self.E_max))

            # Write arrays.
            g.create_dataset('data', data=self.data)
            if self.pseudo_data is not None:
                g.create_dataset('pseudo_data', data=self.pseudo_data)
            g.create_dataset('windows', data=self.windows)
            if self.broaden_poly is not None:
                g.create_dataset('broaden_poly',
                                 data=self.broaden_poly.astype(np.int8))
            if self.curvefit is not None:
                g.create_dataset('curvefit', data=self.curvefit)
