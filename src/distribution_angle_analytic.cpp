#include "openmc/distribution_angle_analytic.h"

#include "openmc/constants.h" // for PI
#include "openmc/error.h"
#include "openmc/math_functions.h"
#include "openmc/random_lcg.h" // Use OpenMC's prn function

#include <cstdlib> // for getenv

namespace openmc {

namespace {
constexpr int N_ENVELOPE_POINTS = 1001;

// Modified-moment closed-form sampler for l<=2 Legendre expansions
// f(mu) = 0.5*(1 + 3*a1*mu) + (5/2)*a2*P2(mu) (with a0 = 1): inverts the
// CDF directly (quadratic branch, chosen with probability ~3*|a1|; cubic
// branch via the trigonometric/Cardano solution otherwise) instead of
// rejecting. Only valid when the truncated series stays a bounded,
// non-negative density -- callers must check that guard themselves.
double sample_from_legendre_modified_moment(
  double w1, double w2, uint64_t* seed)
{
  double w1_abs = std::fabs(w1);

  if (3.0 * w1_abs > prn(seed)) {
    return (-w1_abs +
             std::sqrt(2.0 * w1 * w1 - 2.0 * w1 * w1_abs +
                       4.0 * w1 * prn(seed) * w1_abs)) /
           w1;
  }

  double p = (-5.0 * w2 + 2.0 - 6.0 * w1_abs) / (5.0 * w2);
  double q = (2.0 - 6.0 * w1_abs - 4.0 * prn(seed) * (1.0 - 3.0 * w1_abs)) /
             (5.0 * w2);

  if (p < 0.0) {
    return 2.0 * std::sqrt(-p / 3.0) *
           std::cos(1.0 / 3.0 *
                       std::acos(3.0 * q / (2.0 * p) * std::sqrt(-3.0 / p)) -
                     2.0 * PI / 3.0);
  } else {
    double C =
      std::cbrt(-q / 2.0 + std::sqrt(q * q / 4.0 + p * p * p / 27.0));
    return C - p / (3.0 * C);
  }
}
} // namespace

// Sample mu using Legendre expansion via rejection sampling
double AngleDistributionAnalytic::sample_from_legendre(uint64_t* seed) const
{
  if (distribution_analytic_.empty()) {
    fatal_error("No Legendre coefficients available for angular sampling");
  }

  int n = distribution_analytic_.size() - 1;
  const double* data = distribution_analytic_.data();

  // Experiment mode (env OPENMC_WMP_MODIFIED_MOMENT): for l==2 expansions,
  // use the closed-form CDF inversion above instead of rejection sampling,
  // whenever the inversion is guaranteed valid (see guard below). Falls
  // through to rejection sampling otherwise -- including automatically for
  // n != 2, where no closed form is implemented here.
  static const bool modified_moment = [] {
    bool on = std::getenv("OPENMC_WMP_MODIFIED_MOMENT") != nullptr;
    if (on)
      warning("WMP angle sampling using modified-moment closed-form "
              "inversion where valid (OPENMC_WMP_MODIFIED_MOMENT set).");
    return on;
  }();

  if (modified_moment && n == 2) {
    double w1 = data[1];
    double w2 = data[2];
    double w1_abs = std::fabs(w1);

    // Guard: the closed form is only valid while the l<=2 series stays a
    // bounded, non-negative density; otherwise fall through to rejection
    // sampling below.
    bool needs_rejection =
      (w1_abs >= 1.0 / 3.0) || (w2 > (2.0 - 6.0 * w1_abs) / 5.0);

    if (!needs_rejection) {
      return sample_from_legendre_modified_moment(w1, w2, seed);
    }
  }

  // Find an envelope for the rejection sampling bounding box.
  double p_max = 0.0;
  if (n <= 2) {
    // For expansions truncated at l<=2, the maximum of f(mu) occurs at
    // mu = -1, 0, or 1.
    for (double mu : {-1.0, 0.0, 1.0}) {
      double f = evaluate_legendre(n, data, mu);
      if (f > p_max)
        p_max = f;
    }
  } else {
    // For higher-order expansions, scan a fine grid over mu in [-1, 1]
    // since the maximum may fall elsewhere.
    double dmu = 2.0 / (N_ENVELOPE_POINTS - 1);
    for (int i = 0; i < N_ENVELOPE_POINTS; ++i) {
      double mu = (i == N_ENVELOPE_POINTS - 1) ? 1.0 : -1.0 + i * dmu;
      double f = evaluate_legendre(n, data, mu);
      if (f > p_max)
        p_max = f;
    }
    // Add a margin since the true maximum may fall between scanned points
    p_max *= 1.1;
  }

  if (p_max <= 0.0) {
    p_max = 1.0; // Fallback to avoid an infinite rejection loop
  }

  // Rejection sampling loop
  for (int attempt = 0; attempt < MAX_SAMPLE; ++attempt) {
    double mu = 2.0 * prn(seed) - 1.0;
    double f = evaluate_legendre(n, data, mu);
    if (f > 0.0) {
      double u = prn(seed) * p_max;
      if (u <= f)
        return mu;
    }
  }

  fatal_error("Maximum number of Legendre expansion samples reached in "
              "AngleDistributionAnalytic::sample_from_legendre");
}

} // namespace openmc
