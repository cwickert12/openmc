#include "openmc/distribution_angle_analytic.h"

#include "openmc/error.h"
#include "openmc/math_functions.h"
#include "openmc/random_lcg.h" // Use OpenMC's prn function

namespace openmc {

namespace {
constexpr int N_ENVELOPE_POINTS = 1001;
} // namespace

// Sample mu using Legendre expansion via rejection sampling
double AngleDistributionAnalytic::sample_from_legendre(uint64_t* seed) const
{
  if (distribution_analytic_.empty()) {
    fatal_error("No Legendre coefficients available for angular sampling");
  }

  int n = distribution_analytic_.size() - 1;
  const double* data = distribution_analytic_.data();

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
