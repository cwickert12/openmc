#include "openmc/distribution_angle.h"
#include "openmc/distribution_angle_analytic.h"

#include <filesystem>

#include <cmath> // for abs, copysign

#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"

#include "openmc/endf.h"
#include "openmc/hdf5_interface.h"
#include "openmc/random_lcg.h"
#include "openmc/search.h"
#include "openmc/vector.h" // for vector

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>


namespace openmc {

//==============================================================================
// AngleDistribution implementation
//==============================================================================

AngleDistribution::AngleDistribution(hid_t group)
{
  // Load csv
  //load_csv("Cu63.csv");
  

  // Get incoming energies
  read_dataset(group, "energy", energy_);
  int n_energy = energy_.size();

  // Get outgoing energy distribution data
  vector<int> offsets;
  vector<int> interp;
  hid_t dset = open_dataset(group, "mu");
  read_attribute(dset, "offsets", offsets);
  read_attribute(dset, "interpolation", interp);
  xt::xarray<double> temp;
  read_dataset(dset, temp);
  close_dataset(dset);

  for (int i = 0; i < n_energy; ++i) {
    // Determine number of outgoing energies
    int j = offsets[i];
    int n;
    if (i < n_energy - 1) {
      n = offsets[i + 1] - j;
    } else {
      n = temp.shape()[1] - j;
    }

    // Create and initialize tabular distribution
    auto xs = xt::view(temp, 0, xt::range(j, j + n));
    auto ps = xt::view(temp, 1, xt::range(j, j + n));
    auto cs = xt::view(temp, 2, xt::range(j, j + n));
    vector<double> x {xs.begin(), xs.end()};
    vector<double> p {ps.begin(), ps.end()};
    vector<double> c {cs.begin(), cs.end()};

    // To get answers that match ACE data, for now we still use the tabulated
    // CDF values that were passed through to the HDF5 library. At a later
    // time, we can remove the CDF values from the HDF5 library and
    // reconstruct them using the PDF
    Tabular* mudist =
      new Tabular {x.data(), p.data(), n, int2interp(interp[i]), c.data()};

    distribution_.emplace_back(mudist);
  }

  // Load coefficient angular distribution data

  if (object_exists(group, "energy_coeff")) {
    read_dataset(group, "energy_coeff", energy_coeff_);
    int n_energy_coeff_ = energy_coeff_.size();

    read_dataset(group, "coeffs", coeffs_);

    auto ones = xt::ones<double>({coeffs_.shape()[0], std::size_t{1}});

    coeffs_ = xt::concatenate(xt::xtuple(ones, coeffs_), 1);
  }


}

void AngleDistribution::load_csv(const std::string& filename) {

    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open CSV file: " + filename);

    csv_data_.clear();
    energy_csv_.clear(); 
    std::string line;

    // Skip header row
    if (!std::getline(file, line)) {
        throw std::runtime_error("CSV file is empty: " + filename);
    }

    while (std::getline(file, line)) {
        // Skip empty lines
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        // Skip first column (energy)
        if (!std::getline(ss, cell, ',')) continue;

        try {
            double energy = std::stod(cell);
            energy_csv_.push_back(energy);
        } catch (const std::invalid_argument& e) {
            std::cerr << "[WARNING] Invalid energy value skipped: '" << cell << "' in file " << filename << std::endl;
            continue;  // skip row if energy invalid
        }

        while (std::getline(ss, cell, ',')) {
            // Skip empty cells
            if (cell.empty()) continue;

            try {
                row.push_back(std::stod(cell));
            } catch (const std::invalid_argument& e) {
                std::cerr << "[WARNING] Non-numeric cell skipped: '" << cell << "' in file " << filename << std::endl;
            }
        }

        // Only add row if it has data
        if (!row.empty()) csv_data_.push_back(row);
    }

}


double AngleDistribution::sample(double E, uint64_t* seed) const
{
  // Determine number of incoming energies
  int n_energy_coeff_ = energy_coeff_.size();
  auto n = energy_.size();
  double mu = 0.0;

  // Tabulated path when no Legendre-coefficient table was loaded (standard
  // data files) or when E is above the coefficient table's range.
  if (n_energy_coeff_ == 0 || E > energy_coeff_[n_energy_coeff_-1]){
    // Find energy bin and calculate interpolation factor -- if the energy is
    // outside the range of the tabulated energies, choose the first or last bins
    int i;
    double r;
    if (E < energy_[0]) {
      i = 0;
      r = 0.0;
    } else if (E > energy_[n - 1]) {
      i = n - 2;
      r = 1.0;
    } else {
      i = lower_bound_index(energy_.begin(), energy_.end(), E);
      r = (E - energy_[i]) / (energy_[i + 1] - energy_[i]);
    }

    // Sample between the ith and (i+1)th bin
    if (r > prn(seed))
      ++i;

    mu = distribution_[i]->sample(seed);
  }

  else{
    int i;
    double r;
    if (E < energy_coeff_[0]) {
      i = 0;
      r = 0.0;
    } else if (E > energy_coeff_[n_energy_coeff_ - 1]) {
      i = n - 2;
      r = 1.0;
    } else {
      i = lower_bound_index(energy_coeff_.begin(), energy_coeff_.end(), E);
      r = (E - energy_coeff_[i]) / (energy_coeff_[i + 1] - energy_coeff_[i]);
    }

    // Sample between the ith and (i+1)th bin
    if (r > prn(seed))
      ++i;

    auto row = xt::view(coeffs_, i, xt::all());
    std::vector<double> coeffs_trimmed(row.begin(), row.end()); // remove all trailing zeros 

    constexpr double tol = 1e-14;
    while (!coeffs_trimmed.empty() &&
          std::abs(coeffs_trimmed.back()) < tol) {
      coeffs_trimmed.pop_back();
    }

    AngleDistributionAnalytic dist(coeffs_trimmed);

    
    //std::vector<double> coeffs(row.begin() , row.end());  // exclude energy column 
    
    // Sample i-th distribution
    // double mu = distribution_[i]->sample(seed);

    mu = dist.sample_from_legendre(seed);
  }

  // Make sure mu is in range [-1,1] and return
  if (std::abs(mu) > 1.0)
    mu = std::copysign(1.0, mu);
  
  /*
  std::ofstream myfile;
  myfile.open("mu_vals.txt", std::ios_base::app);
  myfile << E << ", " << mu << std::endl;
  myfile.close();  
  */

  return mu;
}

} // namespace openmc
