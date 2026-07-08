#ifndef OPENMC_DISTRIBUTION_ANGLE_ANALYTIC_H
#define OPENMC_DISTRIBUTION_ANGLE_ANALYTIC_H

#include <vector>
#include <cstdint>

namespace openmc {

class AngleDistributionAnalytic {
public:
    // Constructor: Legendre coefficients
    AngleDistributionAnalytic(const std::vector<double>& coeffs)
        : distribution_analytic_(coeffs) {}
    
    // Sample mu using Legendre expansion
    double sample_from_legendre(uint64_t* seed) const;
    
    // Legendre polynomial
    static double legendreP(int l, double x);

private:
    std::vector<double> distribution_analytic_;
    
    // Derivative of Legendre polynomial (if needed)
    double legendreP_derivative(int l, double x) const;
};

} // namespace openmc

#endif