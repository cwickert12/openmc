//! \file distribution_angle.h
//! Angle distribution dependent on incident particle energy

#ifndef OPENMC_DISTRIBUTION_ANGLE_H
#define OPENMC_DISTRIBUTION_ANGLE_H

#include "hdf5.h"
#include "xtensor/xarray.hpp"

#include "openmc/distribution.h"
#include "openmc/vector.h"
#include <string>

namespace openmc {

//==============================================================================
//! Angle distribution that depends on incident particle energy
//==============================================================================

class AngleDistribution {
public:
  AngleDistribution() = default;
  explicit AngleDistribution(hid_t group);

  //! Sample an angle given an incident particle energy
  //! \param[in] E Particle energy in [eV]
  //! \param[inout] seed pseudorandom number seed pointer
  //! \return Cosine of the angle in the range [-1,1]
  double sample(double E, uint64_t* seed) const;

  //! Determine whether angle distribution is empty
  //! \return Whether distribution is empty
  bool empty() const { return energy_.empty(); }

  // Load Legendre coefficients from a CSV
  

private:
  vector<double> energy_;
  vector<double> energy_csv_;
  vector<double> energy_coeff_;
  vector<unique_ptr<Tabular>> distribution_;
  // CSV data stored in memory
  std::vector<std::vector<double>> csv_data_;
  xt::xarray<double> coeffs_;
  void load_csv(const std::string& filename);
};

} // namespace openmc

#endif // OPENMC_DISTRIBUTION_ANGLE_H
