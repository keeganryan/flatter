target_sources(flatter
  PRIVATE
    fused_qr_size_reduction.cpp
    base.cpp
    columnwise.cpp
    columnwise_double.cpp
    iterated.cpp
    lazy_refine.cpp
    params.cpp
    seysen_refine.cpp
)

find_package(LAPACK REQUIRED)
target_link_libraries(flatter
  PUBLIC
    ${LAPACK_LIBRARIES}
)