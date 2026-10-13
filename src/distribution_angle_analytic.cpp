#include "openmc/distribution_angle_analytic.h"

#include "openmc/constants.h" // for PI
#include "openmc/error.h"
#include "openmc/math_functions.h"
#include "openmc/random_lcg.h" // Use OpenMC's prn function

#include <cstdlib> // for getenv

namespace openmc {

namespace {
constexpr int N_ENVELOPE_POINTS = 1001;

constexpr int MAX_CDF_ITERATIONS = 60;

//! Convergence is tested on the CDF residual |F(mu) - xi|, not on the step
//! |dmu|.  F is dimensionless in [0, 1], so this is an absolute tolerance on
//! the sampled quantile.  Testing the step instead forces a bisection tail --
//! a bracket needs ~47 halvings to reach 1e-14 -- which measured 14 iterations
//! on average against 4.7 for the residual test, at identical accuracy.
constexpr double CDF_TOLERANCE = 1.0e-12;

// Modified-moment sampler for l<=2 Legendre expansions.  Writing
//   f(mu) = A + B*mu + C*mu^2,   A = 0.5 - 1.25*a2, B = 1.5*a1, C = 3.75*a2
// the CDF is
//   F(mu) = A*(mu+1) + B*(mu^2-1)/2 + C*(mu^3+1)/3,   F(-1)=0, F(1)=1
// and a sample is the root of F(mu) = xi.  Solved by Newton iteration
// safeguarded with bisection on [-1, 1]: F is monotone there whenever
// f >= 0, which the caller guarantees, so the root is unique and bracketed
// from the outset.
//
// This replaces an earlier closed-form cubic inversion, which divided by
// 5*a2 and therefore produced inf/NaN as a2 -> 0 (a NaN scattering cosine
// becomes a lost particle), and which selected a fixed root branch that did
// not reproduce a2 for concave (a2 < 0) densities.  A bracketed Newton has
// no branch selection, no cbrt/acos, and no degeneracy at a2 = 0.
double sample_from_legendre_modified_moment(
  double a1, double a2, uint64_t* seed)
{
  const double xi = prn(seed);
  const double A = 0.5 - 1.25 * a2;
  const double B = 1.5 * a1;
  const double C = 3.75 * a2;

  double lo = -1.0;
  double hi = 1.0;
  double mu = 2.0 * xi - 1.0; // uniform starting guess

  for (int i = 0; i < MAX_CDF_ITERATIONS; ++i) {
    double F = A * (mu + 1.0) + 0.5 * B * (mu * mu - 1.0) +
               (C / 3.0) * (mu * mu * mu + 1.0) - xi;
    if (std::fabs(F) < CDF_TOLERANCE)
      break; // mu already inverts the CDF to tolerance

    if (F > 0.0) {
      hi = mu;
    } else {
      lo = mu;
    }

    double f = A + B * mu + C * mu * mu;
    double next = (f > 0.0) ? mu - F / f : 0.5 * (lo + hi);
    if (!(next > lo && next < hi))
      next = 0.5 * (lo + hi); // Newton left the bracket -- bisect instead

    mu = next;
  }

  return std::fmin(1.0, std::fmax(-1.0, mu));
}

// Original closed-form CDF inversion for l==2 expansions, restored
// alongside the Newton inversion above for timing comparison between the
// two.  Splits on which branch of F(mu) = xi solves cleanly: a quadratic
// branch, taken with probability ~3*|a1|, via the quadratic formula, else
// the cubic branch via the trigonometric/Cardano solution.
//
// This is the version the Newton inversion above replaced -- it divides by
// 5*a2 (NaN as a2 -> 0) and picks a fixed root branch that does not track
// a2 for concave (a2 < 0) densities, so it stays valid only on a narrower
// range than the Newton path (see legendre_closed_form_valid below) and
// falls through to rejection sampling -- not to the Newton path -- when
// that guard fails, matching its original behavior.
double sample_from_legendre_modified_moment_closed_form(
  double a1, double a2, uint64_t* seed)
{
  double a1_abs = std::fabs(a1);

  if (3.0 * a1_abs > prn(seed)) {
    return (-a1_abs +
             std::sqrt(2.0 * a1 * a1 - 2.0 * a1 * a1_abs +
                       4.0 * a1 * prn(seed) * a1_abs)) /
           a1;
  }

  double p = (-5.0 * a2 + 2.0 - 6.0 * a1_abs) / (5.0 * a2);
  double q = (2.0 - 6.0 * a1_abs - 4.0 * prn(seed) * (1.0 - 3.0 * a1_abs)) /
             (5.0 * a2);

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

// Validity guard for the closed-form inversion above, exactly as it
// originally shipped: the branch selection and the quadratic-formula
// discriminant only stay well-defined while |a1| < 1/3 and
// a2 <= (2 - 6*|a1|)/5.  Narrower than legendre_density_min's guard for the
// Newton path -- e.g. it rejects some concave (a2 < 0) densities that
// Newton handles fine -- which is expected: it is the guard this inversion
// was originally derived under, not a general non-negativity test.
bool legendre_closed_form_valid(double a1, double a2)
{
  double a1_abs = std::fabs(a1);
  return a1_abs < 1.0 / 3.0 && a2 <= (2.0 - 6.0 * a1_abs) / 5.0;
}

// Minimum of the l<=2 density f(mu) = A + B*mu + C*mu^2 on [-1, 1].  For
// C >= 0 it is convex, so the minimum is at the vertex mu* = -0.2*a1/a2 when
// that is interior (|a1| < 5*a2) and at an endpoint otherwise; for C < 0 it
// is concave and the minimum is always at an endpoint.
double legendre_density_min(double a1, double a2)
{
  double f_min = 0.5 - 1.5 * std::fabs(a1) + 2.5 * a2; // min(f(-1), f(1))
  if (a2 > 0.0 && std::fabs(a1) < 5.0 * a2) {
    double f_vertex = 0.5 - 1.25 * a2 - 0.15 * a1 * a1 / a2;
    if (f_vertex < f_min)
      f_min = f_vertex;
  }
  return f_min;
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
  // use the bracketed-Newton CDF inversion above instead of rejection
  // sampling, whenever the inversion is guaranteed valid (see guard below).
  // Falls through to rejection sampling otherwise -- including
  // automatically for n != 2, where no closed form is implemented here.
  static const bool modified_moment = [] {
    bool on = std::getenv("OPENMC_WMP_MODIFIED_MOMENT") != nullptr;
    if (on)
      warning("WMP angle sampling using modified-moment closed-form "
              "inversion where valid (OPENMC_WMP_MODIFIED_MOMENT set).");
    return on;
  }();

  // Experiment mode (env OPENMC_WMP_MODIFIED_MOMENT_CLOSED_FORM): the
  // original closed-form inversion this Newton path replaced, restored as
  // a separate opt-in for timing comparison between the two.  Distinct env
  // var from OPENMC_WMP_MODIFIED_MOMENT above -- the two implementations
  // solve the same problem differently and should not be conflated.  If
  // both are set, the closed form takes priority.
  static const bool modified_moment_closed_form = [] {
    bool on = std::getenv("OPENMC_WMP_MODIFIED_MOMENT_CLOSED_FORM") != nullptr;
    if (on)
      warning("WMP angle sampling using the original closed-form "
              "modified-moment inversion where valid "
              "(OPENMC_WMP_MODIFIED_MOMENT_CLOSED_FORM set).");
    return on;
  }();

  if ((modified_moment || modified_moment_closed_form) && n == 2) {
    double a1 = data[1];
    double a2 = data[2];

    if (modified_moment_closed_form && legendre_closed_form_valid(a1, a2)) {
      return sample_from_legendre_modified_moment_closed_form(a1, a2, seed);
    }

    // The Newton inversion needs F monotone on [-1, 1], i.e. f >= 0 there.
    // Test that directly rather than with a heuristic on a1/a2: the
    // closed-form guard above misses concave (a2 < 0) densities that dip
    // negative at an endpoint.  If the truncated series is not a valid
    // density, fall through to rejection sampling, which clips at zero
    // instead.
    if (modified_moment && legendre_density_min(a1, a2) >= 0.0) {
      return sample_from_legendre_modified_moment(a1, a2, seed);
    }
  }

  // Find an envelope for the rejection sampling bounding box.
  double p_max = 0.0;
  if (n <= 2) {
    // An l<=2 expansion is the quadratic
    //   f(mu) = (0.5 - 1.25*a2) + 1.5*a1*mu + 3.75*a2*mu^2,
    // so every candidate for its maximum on [-1, 1] is closed form and the
    // envelope needs no evaluate_legendre() call at all.
    //
    // Endpoints: f(+-1) = 0.5 +- 1.5*a1 + 2.5*a2, so the larger of the two is
    // 0.5 + 2.5*a2 + 1.5*|a1|.  mu = 0 is never the maximum -- for a2 > 0 the
    // parabola opens upward and mu = 0 sits near the minimum, and for a2 < 0
    // it is the vertex only when a1 == 0, which the branch below covers
    // exactly -- so it is not worth a candidate.
    double a1 = (n >= 1) ? data[1] : 0.0;
    double a2 = (n >= 2) ? data[2] : 0.0;
    p_max = 0.5 + 2.5 * a2 + 1.5 * std::fabs(a1);

    // For a2 < 0 the parabola opens downward and its maximum is at the vertex
    // mu* = -0.2*a1/a2, interior whenever |a1| < 5*|a2|.  Missing it does not
    // merely slow the rejection loop: acceptance saturates at 1 wherever
    // f > p_max, so the realised density becomes min(f, p_max) -- a clipped
    // peak, i.e. systematically reduced anisotropy.  Closed form via
    // f(mu*) = A - B^2/(4C).
    if (a2 < 0.0 && std::fabs(a1) < 5.0 * std::fabs(a2)) {
      double f_vertex = 0.5 - 1.25 * a2 - 0.15 * a1 * a1 / a2;
      if (f_vertex > p_max)
        p_max = f_vertex;
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
                            
