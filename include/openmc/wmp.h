#ifndef OPENMC_WMP_H
#define OPENMC_WMP_H

#include "hdf5.h"
#include "xtensor/xtensor.hpp"

#include <complex>
#include <cstdint>
#include <string>
#include <tuple>

#include "openmc/array.h"
#include "openmc/vector.h"

namespace openmc {

//========================================================================
// Constants
//========================================================================

// Constants that determine which value to access
constexpr int MP_EA {0}; // Pole
constexpr int MP_RS {1}; // Residue scattering
constexpr int MP_RA {2}; // Residue absorption
constexpr int MP_RF {3}; // Residue fission

// Polynomial fit indices
constexpr int FIT_S {0}; // Scattering
constexpr int FIT_A {1}; // Absorption
constexpr int FIT_F {2}; // Fission

// Multipole HDF5 file version
constexpr array<int, 2> WMP_VERSION {1, 1};

//========================================================================
// Windowed multipole data
//========================================================================

class WindowedMultipole {
public:
  // Types
  struct WindowInfo {
    int index_start;        // Index of starting pole
    int index_end;          // Index of ending pole
    bool broaden_poly;      // Whether to broaden polynomial curvefit
    int pseudo_index_start; // Index of starting pseudopole (-1 if none)
    int pseudo_index_end;   // Index of ending pseudopole (-1 if none)
  };

  // Constructors, destructors
  WindowedMultipole(hid_t group);

  // Methods

  //! \brief Evaluate the windowed multipole equations for cross sections in the
  //! resolved resonance regions
  //!
  //! \param E Incident neutron energy in [eV]
  //! \param sqrtkT Square root of temperature times Boltzmann constant
  //! \return Tuple of elastic scattering, absorption, and fission cross
  //! sections in [b]
  std::tuple<double, double, double> evaluate(double E, double sqrtkT) const;

  //! \brief Evaluates the windowed multipole equations for the derivative of
  //! cross sections in the resolved resonance regions with respect to
  //! temperature.
  //!
  //! \param E Incident neutron energy in [eV]
  //! \param sqrtkT Square root of temperature times Boltzmann constant
  //! \return Tuple of derivatives of elastic scattering, absorption, and
  //!         fission cross sections in [b/K]
  std::tuple<double, double, double> evaluate_deriv(
    double E, double sqrtkT) const;

  //! \brief Evaluate Doppler-broadened Legendre moments from pseudopole data
  //!
  //! Uses the same asymptotic (0K) / Faddeeva-function (T>0) forms as the
  //! physical poles in evaluate(), applied to each pseudopole residue
  //! channel (e.g. sigma_0, sigma_1, sigma_2, ... angular moments).
  //!
  //! \param E Incident neutron energy in [eV]
  //! \param sqrtkT Square root of temperature times Boltzmann constant
  //! \return Vector of Legendre moment values, one per residue channel in
  //!         pseudo_data_. Empty if this nuclide has no pseudopole data.
  vector<double> evaluate_pseudo(double E, double sqrtkT) const;

  //! \brief Sample a scattering cosine from the Doppler-broadened Legendre
  //! moments computed by evaluate_pseudo().
  //!
  //! \param E Incident neutron energy in [eV]
  //! \param sqrtkT Square root of temperature times Boltzmann constant
  //! \param seed Pseudorandom number generator seed
  //! \return Sampled scattering cosine in [-1, 1]. Isotropic if this
  //!         nuclide/window has no pseudopole data.
  double sample_angle(double E, double sqrtkT, uint64_t* seed) const;

  //! \brief Evaluate every channel of a global-pole-table format library
  //!
  //! Global format (AAA-fit libraries): sigma_ch(E) = sum_j Re[r_j /
  //! (sqrt(E) - p_j)] over the window's pool pointer range plus its own
  //! external poles. Channels are the residue columns: [0] elastic
  //! scattering, [1] absorption, [2..] elastic Legendre moments sigma_l.
  //! At T>0 the analytic Faddeeva broadening is applied identically to
  //! every channel; poles with Re(p) == 0 (the 1/v pole) are evaluated
  //! unbroadened (a 1/v cross section is exactly invariant under free-gas
  //! broadening). The low-energy image-kernel correction is NOT
  //! implemented (matters only below ~u < 4.5 xi, i.e. ~20 meV at 3000 K).
  //!
  //! \param E Incident neutron energy in [eV]
  //! \param sqrtkT Square root of temperature times Boltzmann constant
  //! \param xs Output array of n_channels_ channel values in [b]
  void evaluate_global(double E, double sqrtkT, double* xs) const;

  //! Binary-search window lookup for non-uniform (global format) windows
  int find_window(double sqrtE) const;

  // Data members
  std::string name_;               //!< Name of nuclide
  double E_min_;                   //!< Minimum energy in [eV]
  double E_max_;                   //!< Maximum energy in [eV]
  double sqrt_awr_;                //!< Square root of atomic weight ratio
  double inv_spacing_;             //!< 1 / spacing in sqrt(E) space
  int fit_order_;                  //!< Order of the fit
  bool fissionable_;               //!< Is the nuclide fissionable?
  vector<WindowInfo> window_info_; // Information about a window
  xt::xtensor<double, 3>
    curvefit_; // Curve fit coefficients (window, poly order, reaction)
  xt::xtensor<std::complex<double>, 2> data_; //!< Poles and residues
  xt::xtensor<std::complex<double>, 2>
    pseudo_data_; //!< Pseudopoles and residues (e.g. angular moment fits)
  bool has_pseudo_data_ {false}; //!< Whether pseudo_data_ was present

  // Global pole table format (AAA-fit libraries, e.g. Cu63_wmp_global.h5):
  // poles stored once in one alpha-sorted table; windows are contiguous
  // pointer ranges into it (non-uniform in sqrt(E), hence sqrtE_bounds_);
  // each window additionally owns external poles with per-window residues
  // (the curvefit replacement). Residue columns are channels in writer
  // order: [scattering, absorption, sigma_1, sigma_2, ...]. Assumes a
  // non-fissionable nuclide (no fission column).
  bool global_format_ {false};   //!< File uses the global pole table layout
  int n_channels_ {0};           //!< Number of residue columns
  vector<double> sqrtE_bounds_;  //!< Window boundaries in sqrt(E) [n_win+1]
  xt::xtensor<std::complex<double>, 1> poles_;    //!< Pool poles [n_pool]
  xt::xtensor<std::complex<double>, 2> residues_; //!< [n_pool, n_channels]
  xt::xtensor<std::complex<double>, 2> ext_poles_; //!< [n_win, n_ext]
  xt::xtensor<std::complex<double>, 3>
    ext_residues_; //!< [n_win, n_ext, n_channels]

  // Constant data
  static constexpr int MAX_POLY_COEFFICIENTS =
    11; //!< Max order of polynomial fit plus one
};

//========================================================================
// Non-member functions
//========================================================================

//! Check to make sure WMP library data version matches
//!
//! \param[in] file  HDF5 file object
void check_wmp_version(hid_t file);

//! \brief Checks for the existence of a multipole library in the directory and
//! loads it
//!
//! \param[in] i_nuclide  Index in global nuclides array
void read_multipole_data(int i_nuclide);

//==============================================================================
//! Doppler broadens the windowed multipole curvefit.
//!
//! The curvefit is a polynomial of the form a/E + b/sqrt(E) + c + d sqrt(E)...
//!
//! \param E       The energy to evaluate the broadening at
//! \param dopp    sqrt(atomic weight ratio / kT) with kT given in eV
//! \param n       The number of components to the polynomial
//! \param factors The output leading coefficient
//==============================================================================

extern "C" void broaden_wmp_polynomials(
  double E, double dopp, int n, double factors[]);

} // namespace openmc

#endif // OPENMC_WMP_H
