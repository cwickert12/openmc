# Custom test for AngleDistribution with analytical Legendre constructor

add_executable(test_legendre test_legendre.cpp)

# Link against OpenMC library
target_link_libraries(test_legendre PRIVATE openmc)

# Include directories
target_include_directories(test_legendre PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    $ENV{CONDA_PREFIX}/include
)

# Optional: set C++ standard explicitly
target_compile_features(test_legendre PRIVATE cxx_std_17)

# Validation harness for the global-pole-table WMP format
add_executable(test_wmp_global test_wmp_global.cpp)
target_link_libraries(test_wmp_global PRIVATE openmc)
target_include_directories(test_wmp_global PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    $ENV{CONDA_PREFIX}/include
)
target_compile_features(test_wmp_global PRIVATE cxx_std_17)
