target_sources(flatter
  PRIVATE
    base.cpp
    generic.cpp
    triangular.cpp
    orthogonal.cpp
    orthogonal_double.cpp
    params.cpp
    relative_size_reduction.cpp
)

find_package(LAPACK REQUIRED)
target_link_libraries(flatter
  PUBLIC
    ${LAPACK_LIBRARIES}
)