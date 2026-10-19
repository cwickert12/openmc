#include "openmc/wmp.h"

#include "openmc/constants.h"
#include "openmc/cross_sections.h"
#include "openmc/distribution_angle_analytic.h"
#include "openmc/error.h" // for writing messages
#include "openmc/hdf5_interface.h"
#include "openmc/math_functions.h"
#include "openmc/nuclide.h"
#include "openmc/random_lcg.h"

#include <fmt/core.h>

#include <algorithm> // for min
#include <cstdlib>   // for getenv
#include <cmath>

namespace openmc {

//========================================================================
// WindowedeMultipole implementation
//========================================================================

WindowedMultipole::WindowedMultipole(hid_t group)
{
  // Get name of nuclide from group, removing leading '/'
  name_ = object_name(group).substr(1);

  // Global pole table format (AAA-fit libraries): detected by its window
  // pointer datasets. Layout: poles[n_pool] (complex, stored once),
  // residues[n_pool, n_ch] (global, one column per channel), windows are
  // contiguous ranges poles[win_pole_start[w]:win_pole_end[w]] with
  // NON-uniform sqrt(E) boundaries in sqrtE_bounds[n_win+1], plus
  // ext_poles[n_win, n_ext] / ext_residues[n_win, n_ext, n_ch] owned
  // per-window. Channel order (writer convention): scattering, absorption,
  // then elastic Legendre moments sigma_l.
  if (object_exists(group, "win_pole_start")) {
    global_format_ = true;

    double awr;
    read_attribute(group, "AWR", awr);
    sqrt_awr_ = std::sqrt(awr);
    read_attribute(group, "E_min", E_min_);
    read_attribute(group, "E_max", E_max_);

    read_dataset(group, "poles", poles_);
    read_dataset(group, "residues", residues_);
    read_dataset(group, "ext_poles", ext_poles_);
    read_dataset(group, "ext_residues", ext_residues_);
    n_channels_ = residues_.shape()[1];

    xt::xtensor<double, 1> bounds;
    read_dataset(group, "sqrtE_bounds", bounds);
    sqrtE_bounds_.assign(bounds.begin(), bounds.end());

    xt::xtensor<int, 1> win_start, win_end;
    read_dataset(group, "win_pole_start", win_start);
    read_dataset(group, "win_pole_end", win_end);
    int n_windows = win_start.shape()[0];
    if (sqrtE_bounds_.size() != static_cast<size_t>(n_windows) + 1 ||
        ext_poles_.shape()[0] != static_cast<size_t>(n_windows)) {
      fatal_error("Global-format WMP library for " + name_ +
                  " has inconsistent window array shapes.");
    }
    window_info_.resize(n_windows);
    for (int i = 0; i < n_windows; ++i) {
      window_info_[i].index_start = win_start(i);       // already 0-based
      window_info_[i].index_end = win_end(i) - 1;       // end is exclusive
      window_info_[i].broaden_poly = false;             // no curvefit
      window_info_[i].pseudo_index_start = -1;
      window_info_[i].pseudo_index_end = -1;
    }

    fissionable_ = false;
    fit_order_ = -1;                        // no curvefit terms
    // Moment channels beyond [scattering, absorption] drive angle sampling
    // through the same evaluate_pseudo()/sample_angle() path as the
    // pseudopole format.
    has_pseudo_data_ = (n_channels_ > 2);
    return;
  }

  // Read scalar values.
  read_dataset(group, "spacing", inv_spacing_);
  inv_spacing_ = 1.0 / inv_spacing_;
  read_dataset(group, "sqrtAWR", sqrt_awr_);
  read_dataset(group, "E_min", E_min_);
  read_dataset(group, "E_max", E_max_);

  // Read the "data" array.  Use its shape to figure out the number of poles
  // and residue types in this data.
  read_dataset(group, "data", data_);
  int n_residues = data_.shape()[1] - 1;

  // Check to see if this data includes fission residues.
  fissionable_ = (n_residues == 3);

  // Read the "pseudo_data" array, if present. This holds pseudopoles and
  // residues (e.g. angular moment fits) alongside the physical poles in
  // "data".
  has_pseudo_data_ = object_exists(group, "pseudo_data");
  if (has_pseudo_data_) {
    read_dataset(group, "pseudo_data", pseudo_data_);
  }

  // Read the "windows" array and use its shape to figure out the number of
  // windows. When pseudo_data is present, windows has 4 columns
  // (pole_start, pole_end, pseudo_start, pseudo_end); otherwise 2
  // (pole_start, pole_end).
  xt::xtensor<int, 2> windows;
  read_dataset(group, "windows", windows);
  int n_windows = windows.shape()[0];
  bool has_pseudo_windows = windows.shape()[1] >= 4;
  windows -= 1; // Adjust to 0-based indices

  // Read the "broaden_poly" arrays.
  xt::xtensor<bool, 1> broaden_poly;
  read_dataset(group, "broaden_poly", broaden_poly);
  if (n_windows != broaden_poly.shape()[0]) {
    fatal_error("broaden_poly array shape is not consistent with the windows "
                "array shape in WMP library for " +
                name_ + ".");
  }

  // Read the "curvefit" array.
  read_dataset(group, "curvefit", curvefit_);
  if (n_windows != curvefit_.shape()[0]) {
    fatal_error("curvefit array shape is not consistent with the windows "
                "array shape in WMP library for " +
                name_ + ".");
  }
  fit_order_ = curvefit_.shape()[1] - 1;

  // pseudo_data and the 4-column form of windows must appear together.
  if (has_pseudo_data_ != has_pseudo_windows) {
    fatal_error("WMP library for " + name_ +
                " has pseudo_data without 4-column windows, or vice versa.");
  }

  // Check the code is compiling to work with sufficiently high fit order
  if (fit_order_ + 1 > MAX_POLY_COEFFICIENTS) {
    fatal_error(fmt::format(
      "Need to compile with WindowedMultipole::MAX_POLY_COEFFICIENTS = {}",
      fit_order_ + 1));
  }

  // Move window information into a vector
  window_info_.resize(n_windows);
  for (int i = 0; i < n_windows; ++i) {
    window_info_[i].index_start = windows(i, 0);
    window_info_[i].index_end = windows(i, 1);
    window_info_[i].broaden_poly = broaden_poly[i];
    if (has_pseudo_windows) {
      window_info_[i].pseudo_index_start = windows(i, 2);
      window_info_[i].pseudo_index_end = windows(i, 3);
    } else {
      window_info_[i].pseudo_index_start = -1;
      window_info_[i].pseudo_index_end = -1;
    }
  }
}

int WindowedMultipole::find_window(double sqrtE) const
{
  auto it =
    std::upper_bound(sqrtE_bounds_.begin(), sqrtE_bounds_.end(), sqrtE);
  int i = static_cast<int>(it - sqrtE_bounds_.begin()) - 1;
  return std::min(std::max(i, 0),
    static_cast<int>(window_info_.size()) - 1);
}

namespace {

//! Rational approximation of the Faddeeva function w(z) for Im(z) > 0: a
//! sixth-order shifted rational N(z + 1.5i) / D(z + 1.5i) with D even and the
//! leading numerator coefficient pinned to i/sqrt(pi). From B. Forget, J. Yu
//! and G. Ridley, "Direct Doppler broadening of windowed multipole cross
//! sections with a rational approximation of the Faddeeva function," Proc.
//! PHYSOR 2022; coefficients as fitted in B. Forget's wmp-v2 OpenMC branch over
//! |Re z| <= 600, 1e-4 <= Im z <= 1200 (max relative error 1.6e-5). Only
//! called here for |z| < FAR_Z, well inside that domain.
inline void faddeeva_rational(double zr, double zi, double& wr, double& wi)
{
  zi += 1.5; // shift
  double nr = 0.0, ni = 0.5641895835477563, t;
  t = nr * zr - ni * zi - 0.8463988774687549;
  ni = nr * zi + ni * zr;
  nr = t;
  t = nr * zr - ni * zi;
  ni = nr * zi + ni * zr - 4.262789761626753;
  nr = t;
  t = nr * zr - ni * zi + 5.57957343594273;
  ni = nr * zi + ni * zr;
  nr = t;
  t = nr * zr - ni * zi;
  ni = nr * zi + ni * zr + 8.876835541419577;
  nr = t;
  t = nr * zr - ni * zi - 8.2804011586059;
  ni = nr * zi + ni * zr;
  nr = t;
  double zzr = zr * zr - zi * zi, zzi = 2.0 * zr * zi;
  double dnr = 1.0, dni = 0.0;
  t = dnr * zzr - dni * zzi - 5.810043673216703;
  dni = dnr * zzi + dni * zzr;
  dnr = t;
  t = dnr * zzr - dni * zzi + 6.844707994387722;
  dni = dnr * zzi + dni * zzr;
  dnr = t;
  t = dnr * zzr - dni * zzi - 0.901452218111202;
  dni = dnr * zzi + dni * zzr;
  dnr = t;
  double dm = 1.0 / (dnr * dnr + dni * dni);
  wr = (nr * dnr + ni * dni) * dm;
  wi = (ni * dnr - nr * dni) * dm;
}

//! Poles farther than this many Doppler widths from sqrt(E) take the
//! asymptotic form in broadened_kernel().
constexpr double FAR_Z = 5.0;

//! Doppler-broadened kernel of one pole, already divided by E:
//!   b = (1/E) [ (u + p) + p^2 (i sqrt(pi)/xi) w(z) ],  z = (p - u)/xi,
//! with u = sqrt(E), returned as (br, bi) so that a channel with residue r
//! contributes Re(r b). Written in real arithmetic: std::complex products and
//! quotients go through NaN-checking library calls.
//!
//! - Re(p) == 0 (the 1/v pole) is invariant under free-gas broadening:
//!   b = 1/(u - p).
//! - |z| >= FAR_Z: substituting w(z) ~ i/(sqrt(pi) z) (1 + 1/(2z^2) +
//!   3/(4z^4) + ...) analytically gives b = [1 + (p/u)^2 c] / (u - p) with
//!   c = t/2 + 3t^2/4 + 15t^3/8 + 105t^4/16, t = 1/z^2. Evaluating the
//!   original form with any approximate w(z) instead cancels catastrophically
//!   for far poles (|z| reaches ~1e4 for the wide windows' external poles), and
//!   this form needs no w(z) at all. Truncation error ~3e-6 at |z| = FAR_Z.
//! - Otherwise w(z) comes from faddeeva_rational().
inline void broadened_kernel(double pr, double pi, double u, double invE,
  double inv_xi, double& br, double& bi)
{
  const double dr = u - pr; // u - p = dr - i pi
  const double inv_d2 = 1.0 / (dr * dr + pi * pi);
  if (std::abs(pr) < 1e-12) {
    br = dr * inv_d2;
    bi = pi * inv_d2;
    return;
  }

  const double zr = -dr * inv_xi;
  const double zi = pi * inv_xi;
  const double z2r = zr * zr - zi * zi, z2i = 2.0 * zr * zi;
  const double az2 = zr * zr + zi * zi;
  if (az2 >= FAR_Z * FAR_Z) {
    // t = 1/z^2, c = t (1/2 + t (3/4 + t (15/8 + t 105/16)))
    const double inv = 1.0 / (az2 * az2);
    const double tr = z2r * inv, ti = -z2i * inv;
    double cr = 105.0 / 16.0, ci = 0.0, s;
    s = cr * tr - ci * ti + 15.0 / 8.0;
    ci = cr * ti + ci * tr;
    cr = s;
    s = cr * tr - ci * ti + 0.75;
    ci = cr * ti + ci * tr;
    cr = s;
    s = cr * tr - ci * ti + 0.5;
    ci = cr * ti + ci * tr;
    cr = s;
    s = cr * tr - ci * ti;
    ci = cr * ti + ci * tr;
    cr = s;
    // num = 1 + (p^2/E) c
    const double p2r = (pr * pr - pi * pi) * invE, p2i = 2.0 * pr * pi * invE;
    const double nr = 1.0 + p2r * cr - p2i * ci, ni = p2r * ci + p2i * cr;
    // b = num / (u - p) = num (dr + i pi) / |u - p|^2
    br = (nr * dr - ni * pi) * inv_d2;
    bi = (nr * pi + ni * dr) * inv_d2;
    return;
  }

  double wr, wi;
  faddeeva_rational(zr, zi, wr, wi);
  const double sq = SQRT_PI * inv_xi;
  const double gr = -sq * wi, gi = sq * wr; // i sqrt(pi)/xi w(z)
  const double p2r = pr * pr - pi * pi, p2i = 2.0 * pr * pi;
  const double hr = p2r * gr - p2i * gi, hi = p2r * gi + p2i * gr;
  br = (u + pr + hr) * invE;
  bi = (pi + hi) * invE;
}

} // namespace

void WindowedMultipole::evaluate_global(
  double E, double sqrtkT, double* xs) const
{
  using namespace std::complex_literals;

  double sqrtE = std::sqrt(E);
  double invE = 1.0 / E;
  int i_window = find_window(sqrtE);
  const auto& window {window_info_[i_window]};
  int n_ext = ext_poles_.shape()[1];

  for (int c = 0; c < n_channels_; ++c)
    xs[c] = 0.0;

  if (sqrtkT == 0.0) {
    // 0K: plain pole sum, sigma_ch = Re[r / (sqrt(E) - p)]
    auto add_pole = [&](std::complex<double> p, auto residue) {
      std::complex<double> c_temp = 1.0 / (sqrtE - p);
      for (int c = 0; c < n_channels_; ++c)
        xs[c] += (residue(c) * c_temp).real();
    };
    for (int j = window.index_start; j <= window.index_end; ++j) {
      add_pole(poles_(j), [&](int c) { return residues_(j, c); });
    }
    for (int k = 0; k < n_ext; ++k) {
      add_pole(ext_poles_(i_window, k),
        [&](int c) { return ext_residues_(i_window, k, c); });
    }
  } else {
    // T>0: analytic Faddeeva broadening,
    //   sigma_T(E) = (1/E) Re[ r*(u+p) + r*p^2*(i sqrt(pi) dopp)*w((p-u)*dopp) ]
    // with u = sqrt(E), dopp = 1/xi = sqrt(AWR)/sqrtkT. The 1/v pole
    // (Re p == 0) is exactly invariant under free-gas broadening and is
    // evaluated unbroadened. NOTE: the low-energy image-kernel correction
    // (only relevant below u < 4.5 xi, ~20 meV at 3000 K) is not applied.
    double dopp = sqrt_awr_ / sqrtkT;

    // Experiment mode (env OPENMC_WMP_FADDEEVA_EXACT): evaluate every pole
    // with the exact Faddeeva function instead of broadened_kernel(), for
    // validating and timing the latter.
    static const bool exact = [] {
      bool on = std::getenv("OPENMC_WMP_FADDEEVA_EXACT") != nullptr;
      if (on)
        warning("Global-format WMP broadening uses the exact Faddeeva "
                "function (OPENMC_WMP_FADDEEVA_EXACT set).");
      return on;
    }();

    auto add_pole = [&](std::complex<double> p, auto residue) {
      double br, bi;
      if (exact) {
        std::complex<double> base;
        if (std::abs(p.real()) < 1e-12) {
          base = E / (sqrtE - p); // unbroadened, pre-multiplied by E
        } else {
          std::complex<double> w_val = faddeeva((p - sqrtE) * dopp);
          base = (sqrtE + p) + p * p * (1.0i * SQRT_PI * dopp) * w_val;
        }
        base *= invE;
        br = base.real();
        bi = base.imag();
      } else {
        broadened_kernel(p.real(), p.imag(), sqrtE, invE, dopp, br, bi);
      }
      for (int c = 0; c < n_channels_; ++c) {
        std::complex<double> r = residue(c);
        xs[c] += r.real() * br - r.imag() * bi;
      }
    };
    for (int j = window.index_start; j <= window.index_end; ++j) {
      add_pole(poles_(j), [&](int c) { return residues_(j, c); });
    }
    for (int k = 0; k < n_ext; ++k) {
      add_pole(ext_poles_(i_window, k),
        [&](int c) { return ext_residues_(i_window, k, c); });
    }
  }
}

std::tuple<double, double, double> WindowedMultipole::evaluate(
  double E, double sqrtkT) const
{
  using namespace std::complex_literals;

  if (global_format_) {
    int n_moments;
    return evaluate(E, sqrtkT, nullptr, 0, n_moments);
  }

  // ==========================================================================
  // Bookkeeping

  // Define some frequently used variables.
  double sqrtE = std::sqrt(E);
  double invE = 1.0 / E;

  // Locate window containing energy
  int i_window = std::min(window_info_.size() - 1,
    static_cast<size_t>((sqrtE - std::sqrt(E_min_)) * inv_spacing_));
  const auto& window {window_info_[i_window]};

  // Initialize the ouptut cross sections
  double sig_s = 0.0;
  double sig_a = 0.0;
  double sig_f = 0.0;

  // ==========================================================================
  // Add the contribution from the curvefit polynomial.

  if (sqrtkT > 0.0 && window.broaden_poly) {
    // Broaden the curvefit.
    double dopp = sqrt_awr_ / sqrtkT;
    array<double, MAX_POLY_COEFFICIENTS> broadened_polynomials;
    broaden_wmp_polynomials(
      E, dopp, fit_order_ + 1, broadened_polynomials.data());
    for (int i_poly = 0; i_poly < fit_order_ + 1; ++i_poly) {
      sig_s +=
        curvefit_(i_window, i_poly, FIT_S) * broadened_polynomials[i_poly];
      sig_a +=
        curvefit_(i_window, i_poly, FIT_A) * broadened_polynomials[i_poly];
      if (fissionable_) {
        sig_f +=
          curvefit_(i_window, i_poly, FIT_F) * broadened_polynomials[i_poly];
      }
    }
  } else {
    // Evaluate as if it were a polynomial
    double temp = invE;
    for (int i_poly = 0; i_poly < fit_order_ + 1; ++i_poly) {
      sig_s += curvefit_(i_window, i_poly, FIT_S) * temp;
      sig_a += curvefit_(i_window, i_poly, FIT_A) * temp;
      if (fissionable_) {
        sig_f += curvefit_(i_window, i_poly, FIT_F) * temp;
      }
      temp *= sqrtE;
    }
  }

  // ==========================================================================
  // Add the contribution from the poles in this window.

  if (sqrtkT == 0.0) {
    // If at 0K, use asymptotic form.
    for (int i_pole = window.index_start; i_pole <= window.index_end;
         ++i_pole) {
      std::complex<double> psi_chi = -1.0i / (data_(i_pole, MP_EA) - sqrtE);
      std::complex<double> c_temp = psi_chi * invE;
      sig_s += (data_(i_pole, MP_RS) * c_temp).real();
      sig_a += (data_(i_pole, MP_RA) * c_temp).real();
      if (fissionable_) {
        sig_f += (data_(i_pole, MP_RF) * c_temp).real();
      }
    }
  } else {
    // At temperature, use Faddeeva function-based form.
    double dopp = sqrt_awr_ / sqrtkT;
    for (int i_pole = window.index_start; i_pole <= window.index_end;
         ++i_pole) {
      std::complex<double> z = (sqrtE - data_(i_pole, MP_EA)) * dopp;
      std::complex<double> w_val = faddeeva(z) * dopp * invE * SQRT_PI;
      sig_s += (data_(i_pole, MP_RS) * w_val).real();
      sig_a += (data_(i_pole, MP_RA) * w_val).real();
      if (fissionable_) {
        sig_f += (data_(i_pole, MP_RF) * w_val).real();
      }
    }
  }

  return std::make_tuple(sig_s, sig_a, sig_f);
}

vector<double> WindowedMultipole::evaluate_pseudo(double E, double sqrtkT) const
{
  using namespace std::complex_literals;

  if (!has_pseudo_data_)
    return {};

  // Global format: the moments are residue columns [2..] on the same poles.
  // Return [sigma_0, sigma_1, sigma_2, ...] so that sample_angle()'s
  // normalization a_l = moments[l]/moments[0] yields the Legendre
  // coefficients (channel 0 is the elastic cross section sigma_0).
  if (global_format_) {
    vector<double> xs(n_channels_);
    evaluate_global(E, sqrtkT, xs.data());
    vector<double> moments(n_channels_ - 1);
    moments[0] = xs[0]; // sigma_0 = elastic scattering
    for (int c = 2; c < n_channels_; ++c)
      moments[c - 1] = xs[c];
    return moments;
  }

  // Define some frequently used variables.
  double sqrtE = std::sqrt(E);
  double invE = 1.0 / E;

  // Locate window containing energy
  int i_window = std::min(window_info_.size() - 1,
    static_cast<size_t>((sqrtE - std::sqrt(E_min_)) * inv_spacing_));
  const auto& window {window_info_[i_window]};

  int n_moments = pseudo_data_.shape()[1] - 1;
  vector<double> moments(n_moments, 0.0);

  // Windows are not required to contain any pseudopoles.
  if (window.pseudo_index_start < 0)
    return moments;

  if (sqrtkT == 0.0) {
    // If at 0K, use asymptotic form.
    for (int i_pole = window.pseudo_index_start;
         i_pole <= window.pseudo_index_end; ++i_pole) {
      std::complex<double> psi_chi =
        -1.0i / (pseudo_data_(i_pole, 0) - sqrtE);
      std::complex<double> c_temp = psi_chi * invE;
      for (int i_mom = 0; i_mom < n_moments; ++i_mom) {
        moments[i_mom] += (pseudo_data_(i_pole, i_mom + 1) * c_temp).real();
      }
    }
  } else {
    // At temperature, use Faddeeva function-based form.
    double dopp = sqrt_awr_ / sqrtkT;
    for (int i_pole = window.pseudo_index_start;
         i_pole <= window.pseudo_index_end; ++i_pole) {
      std::complex<double> z = (sqrtE - pseudo_data_(i_pole, 0)) * dopp;
      std::complex<double> w_val = faddeeva(z) * dopp * invE * SQRT_PI;
      for (int i_mom = 0; i_mom < n_moments; ++i_mom) {
        moments[i_mom] += (pseudo_data_(i_pole, i_mom + 1) * w_val).real();
      }
    }
  }

  return moments;
}

std::tuple<double, double, double> WindowedMultipole::evaluate(double E,
  double sqrtkT, double* moments, int max_moments, int& n_moments) const
{
  n_moments = 0;
  if (!global_format_)
    return evaluate(E, sqrtkT);

  vector<double> xs(n_channels_);
  evaluate_global(E, sqrtkT, xs.data());

  // Same layout evaluate_pseudo() returns: [sigma_0, sigma_1, ...], where
  // sigma_0 is the elastic channel and channel 1 (absorption) is skipped.
  int n = n_channels_ - 1;
  if (moments && has_pseudo_data_ && n <= max_moments) {
    moments[0] = xs[0];
    for (int c = 2; c < n_channels_; ++c)
      moments[c - 1] = xs[c];
    n_moments = n;
  }
  return std::make_tuple(xs[0], xs[1], 0.0);
}

double WindowedMultipole::sample_angle(double E, double sqrtkT,
  uint64_t* seed, const double* moments, int n_moments) const
{
  // Experiment mode (env OPENMC_WMP_ANGLE_0K): sample scattering angles
  // from UNBROADENED (0 K) Legendre moments while cross sections remain
  // Doppler broadened -- mimics standard production libraries, where
  // BROADR broadens File 3 but File 4 angular data is never broadened.
  static const bool angle_0k = [] {
    bool on = std::getenv("OPENMC_WMP_ANGLE_0K") != nullptr;
    if (on)
      warning("WMP angle sampling forced to 0 K moments "
              "(OPENMC_WMP_ANGLE_0K set); cross sections remain broadened.");
    return on;
  }();
  // Reuse moments evaluated with the cross sections when they are for the
  // temperature wanted here; the 0 K experiment mode needs its own.
  vector<double> evaluated;
  if (!moments || n_moments == 0 || (angle_0k && sqrtkT != 0.0)) {
    evaluated = evaluate_pseudo(E, angle_0k ? 0.0 : sqrtkT);
    moments = evaluated.data();
    n_moments = static_cast<int>(evaluated.size());
  }

  // No pseudopole data for this nuclide, or none in this window: fall back
  // to isotropic scattering.
  if (n_moments == 0 || moments[0] <= 0.0) {
    return 2.0 * prn(seed) - 1.0;
  }

  // Normalize so that a_0 = 1, matching the convention expected by
  // AngleDistributionAnalytic / evaluate_legendre.
  vector<double> a(n_moments);
  for (int i = 0; i < n_moments; ++i) {
    a[i] = moments[i] / moments[0];
  }

  AngleDistributionAnalytic dist(a);
  return dist.sample_from_legendre(seed);
}

std::tuple<double, double, double> WindowedMultipole::evaluate_deriv(
  double E, double sqrtkT) const
{
  if (global_format_) {
    fatal_error("Windowed multipole temperature derivatives are not "
                "implemented for global-pole-table format libraries.");
  }

  // ==========================================================================
  // Bookkeeping

  // Define some frequently used variables.
  double sqrtE = std::sqrt(E);
  double invE = 1.0 / E;
  double T = sqrtkT * sqrtkT / K_BOLTZMANN;

  if (sqrtkT == 0.0) {
    fatal_error("Windowed multipole temperature derivatives are not implemented"
                " for 0 Kelvin cross sections.");
  }

  // Locate us
  int i_window = (sqrtE - std::sqrt(E_min_)) * inv_spacing_;
  const auto& window {window_info_[i_window]};

  // Initialize the ouptut cross sections.
  double sig_s = 0.0;
  double sig_a = 0.0;
  double sig_f = 0.0;

  // TODO Polynomials: Some of the curvefit polynomials Doppler broaden so
  // rigorously we should be computing the derivative of those.  But in
  // practice, those derivatives are only large at very low energy and they
  // have no effect on reactor calculations.

  // ==========================================================================
  // Add the contribution from the poles in this window.

  double dopp = sqrt_awr_ / sqrtkT;
  for (int i_pole = window.index_start; i_pole <= window.index_end; ++i_pole) {
    std::complex<double> z = (sqrtE - data_(i_pole, MP_EA)) * dopp;
    std::complex<double> w_val = -invE * SQRT_PI * 0.5 * w_derivative(z, 2);
    sig_s += (data_(i_pole, MP_RS) * w_val).real();
    sig_a += (data_(i_pole, MP_RA) * w_val).real();
    if (fissionable_) {
      sig_f += (data_(i_pole, MP_RF) * w_val).real();
    }
  }
  double norm = -0.5 * sqrt_awr_ / std::sqrt(K_BOLTZMANN) * std::pow(T, -1.5);
  sig_s *= norm;
  sig_a *= norm;
  sig_f *= norm;

  return std::make_tuple(sig_s, sig_a, sig_f);
}

//========================================================================
// Non-member functions
//========================================================================

void check_wmp_version(hid_t file)
{
  if (attribute_exists(file, "version")) {
    array<int, 2> version;
    read_attribute(file, "version", version);
    if (version[0] != WMP_VERSION[0]) {
      fatal_error(fmt::format(
        "WMP data format uses version {}.{} whereas your installation of "
        "OpenMC expects version {}.x data.",
        version[0], version[1], WMP_VERSION[0]));
    }
  } else {
    fatal_error(fmt::format("WMP data does not indicate a version. Your "
                            "installation of OpenMC expects version {}x data.",
      WMP_VERSION[0]));
  }
}

void read_multipole_data(int i_nuclide)
{
  // Look for WMP data in cross_sections.xml
  const auto& nuc {data::nuclides[i_nuclide]};
  auto it = data::library_map.find({Library::Type::wmp, nuc->name_});

  // If no WMP library for this nuclide, just return
  if (it == data::library_map.end())
    return;

  // Check if WMP library exists
  int idx = it->second;
  std::string& filename = data::libraries[idx].path_;

  // Display message
  write_message(6, "Reading {} WMP data from {}", nuc->name_, filename);

  // Open file and make sure version is sufficient
  hid_t file = file_open(filename, 'r');
  check_wmp_version(file);

  // Read nuclide data from HDF5
  hid_t group = open_group(file, nuc->name_.c_str());
  nuc->multipole_ = make_unique<WindowedMultipole>(group);
  close_group(group);
  file_close(file);
}

void broaden_wmp_polynomials(double E, double dopp, int n, double factors[])
{
  // Broadening of polynomials follows procedure outlined in C. Josey, P. Ducru,
  // B. Forget, and K. Smith, "Windowed multipole for cross section Doppler
  // broadening," J. Comput. Phys., 307, 715-727 (2016).
  // https://doi.org/10.1016/j.jcp.2015.08.013

  // Factors is already pre-allocated
  double sqrtE = std::sqrt(E);
  double beta = sqrtE * dopp;
  double half_inv_dopp2 = 0.5 / (dopp * dopp);
  double quarter_inv_dopp4 = half_inv_dopp2 * half_inv_dopp2;

  double erf_beta;    // error function of beta
  double exp_m_beta2; // exp(-beta**2)
  if (beta > 6.0) {
    // Save time, ERF(6) is 1 to machine precision.
    // beta/sqrtpi*exp(-beta**2) is also approximately 1 machine epsilon.
    erf_beta = 1.;
    exp_m_beta2 = 0.;
  } else {
    erf_beta = std::erf(beta);
    exp_m_beta2 = std::exp(-beta * beta);
  }

  // Assume that, for sure, we'll use a second order (1/E, 1/V, const)
  // fit, and no less.
  factors[0] = erf_beta / E;
  factors[1] = 1. / sqrtE;
  factors[2] =
    factors[0] * (half_inv_dopp2 + E) + exp_m_beta2 / (beta * SQRT_PI);
  if (n > 3)
    factors[3] = factors[1] * (E + 3.0 * half_inv_dopp2);

  // Perform recursive broadening of high order components (Eq. 16)
  for (int i = 1; i < n - 3; i++) {
    double ip1_dbl = i + 1;
    factors[i + 3] =
      -factors[i - 1] * (ip1_dbl - 1.) * ip1_dbl * quarter_inv_dopp4 +
      factors[i + 1] * (E + (1. + 2. * ip1_dbl) * half_inv_dopp2);
  }
}

} // namespace openmc
