#define _USE_MATH_DEFINES
#include "openmc/distribution_angle_analytic.h"
#include "openmc/random_lcg.h"  // Use OpenMC's prn function
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <iomanip>

namespace openmc {

// Sample mu using Legendre expansion
double AngleDistributionAnalytic::sample_from_legendre(uint64_t* seed) const {
    if (distribution_analytic_.empty()) {
        throw std::runtime_error("No Legendre coefficients available");
    }

    double w1_abs = std::fabs(distribution_analytic_[1]);
    
    //if (w1_abs >= 1.0/3.0 | distribution_analytic_[2] > (2.0-6.0*w1_abs)/5.0) {
    if (true) {
        // Find p_max for rejection sampling
        double p_max = 0.0;

        for (double mu_test = -1.0; mu_test <= 1.0; mu_test += 1) {
            double p = 0.0;
            for (size_t l = 0; l < distribution_analytic_.size(); ++l) {
                p += distribution_analytic_[l] * legendreP(l, mu_test);
            }

            if (p > p_max) p_max = p;
        }
        
        // Safety check
        if (p_max <= 0.0) {
            p_max = 1.0;  // Fallback to avoid infinite loop
        }

        // Rejection sampling loop
        int attempts = 0;
        while (attempts < 10000) {  // Safety limit
            double mu = prn(seed) * 2.0 - 1.0;  // Use OpenMC's prn function
            double y  = prn(seed) * p_max;
            
            double p = 0.0;
            for (size_t l = 0; l < distribution_analytic_.size(); ++l) {
                p += distribution_analytic_[l] * legendreP(l, mu);
            }
            
            if (y <= p && p >= 0.0) {
                return mu;
            }
            attempts++;

        }
        std::cout << distribution_analytic_[0] << " " << distribution_analytic_[1] << " " << distribution_analytic_[-1] << " " << "\n";
        std::cout << "revert to isotropic" << "\n"; 
        // Fallback to isotropic if rejection sampling fails
        return prn(seed) * 2.0 - 1.0;
    }

    else {
        double w0 = 1.0; 
        double w1 = distribution_analytic_[1];
        double w2 = distribution_analytic_[2];

        if (3.0*w1_abs > prn(seed)) {
            return (-1.0*w1_abs+std::sqrt(2*w1*w1-2*w1*w1_abs+4.0*w1*prn(seed)*w1_abs))/w1;
        }

        else {
            double p = 1/(5.0*w2)*(-5.0*w2+2.0-6.0*w1_abs);
            double q = 1/(5.0*w2)*(2.0-6.0*w1_abs-4.0*prn(seed)*(1.0-3.0*w1_abs));

            if (p < 0) { 
                return 2.0*std::sqrt(-p/3.0)*std::cos(1.0/3.0*std::acos(3.0*q/(2.0*p)*std::sqrt(-3.0/p))-2.0*M_PI/3.0);
            }

            else {
                double C = std::cbrt(-q/2.0+std::sqrt(q*q/4.0+p*p*p/27.0));

                return C-p/(3.0*C);
            }
        }

    }

}  



// Legendre polynomial using recurrence relation
double AngleDistributionAnalytic::legendreP(int l, double x) {
    if (l == 0) return 0.5*1.0;
    if (l == 1) return 1.5*x;
    if (l == 2) return 2.5*0.5*(3*x*x-1);
    
}

// Derivative of Legendre polynomial (if needed)
double AngleDistributionAnalytic::legendreP_derivative(int l, double x) const {
    if (l == 0) return 0.0;
    if (std::fabs(x) >= 1.0) {
        // Handle edge cases at x = ±1
        if (x >= 1.0) return l * (l + 1.0) / 2.0 * (l % 2 == 0 ? 1.0 : -1.0);
        if (x <= -1.0) return l * (l + 1.0) / 2.0 * (l % 2 == 0 ? -1.0 : 1.0);
    }
    
    return l / (x*x - 1.0) * (x * legendreP(l, x) - legendreP(l-1, x));
}

} // namespace openmc