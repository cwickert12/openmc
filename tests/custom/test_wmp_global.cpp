// Validation harness for the global-pole-table WMP format: loads a library
// through WindowedMultipole and prints channel values at requested energies
// and temperature, for diffing against the Python WMPLibrary evaluator.
//
// Usage: test_wmp_global <wmp.h5> <nuclide> <sqrtkT> <E1> [E2 ...]
// Output: one line per energy: E sigma_s sigma_a [moments...]
#include "openmc/hdf5_interface.h"
#include "openmc/wmp.h"

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char* argv[])
{
  if (argc < 5) {
    std::fprintf(
      stderr, "usage: %s <wmp.h5> <nuclide> <sqrtkT> <E1> [E2 ...]\n", argv[0]);
    return 1;
  }
  std::string path = argv[1];
  std::string nuclide = argv[2];
  double sqrtkT = std::atof(argv[3]);

  hid_t file = openmc::file_open(path, 'r');
  hid_t group = openmc::open_group(file, nuclide.c_str());
  openmc::WindowedMultipole wmp(group);
  openmc::close_group(group);
  openmc::file_close(file);

  std::printf("# %s: global_format=%d n_channels=%d windows=%zu "
              "E=[%g, %g] has_pseudo=%d\n",
    wmp.name_.c_str(), wmp.global_format_, wmp.n_channels_,
    wmp.window_info_.size(), wmp.E_min_, wmp.E_max_, wmp.has_pseudo_data_);

  for (int i = 4; i < argc; ++i) {
    double E = std::atof(argv[i]);
    auto [sig_s, sig_a, sig_f] = wmp.evaluate(E, sqrtkT);
    std::printf("%.10e %.10e %.10e", E, sig_s, sig_a);
    auto moments = wmp.evaluate_pseudo(E, sqrtkT);
    for (std::size_t m = 1; m < moments.size(); ++m) {
      std::printf(" %.10e", moments[m]);
    }
    std::printf("\n");
  }
  return 0;
}
