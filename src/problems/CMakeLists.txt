target_sources(flatter
  PRIVATE
  problem.cpp
)

add_subdirectory(fused_qr_sizered)
add_subdirectory(matrix_multiplication)
add_subdirectory(qr_factorization)
add_subdirectory(lattice_reduction)
add_subdirectory(size_reduction)
add_subdirectory(relative_size_reduction)