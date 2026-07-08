#ifndef OPENMC_DISTRIBUTION_ANGLE_ANALYTIC_H
#define OPENMC_DISTRIBUTION_ANGLE_ANALYTIC_H

#include <vector>
#include <cstdint>

namespace openmc {

//! Samples a scattering cosine from a normalized Legendre moment expansion
//! f(mu) = sum_l (l+0.5) * a_l * P_l(mu), where a_0 = 1. Supports arbitrary
//! order via rejection sampling with a numerically-scanned envelope.
class AngleDistributionAnalytic {
public:
  // Constructor: normalized Legendre moments a_l (a_0 = 1, without the
  // (l+0.5) factor), ordered by increasing l
  AngleDistributionAnalytic(const std::vector<double>& coeffs)
    : distribution_analytic_(coeffs)
  {}

  // Sample mu using Legendre expansion via rejection sampling
  double sample_from_legendre(uint64_t* seed) const;

private:
  std::vector<double> distribution_analytic_;
};

} // namespace openmc

#endif