find_package(Eigen3 3.3 REQUIRED NO_MODULE)

target_sources(flatter
  PRIVATE
    base.cpp
    blocked.cpp
    householder_mpfr.cpp
    eigen_impl.cpp
    qr_factorization.cpp
    threaded.cpp
)

target_link_libraries(flatter
  PUBLIC
    Eigen3::Eigen
)