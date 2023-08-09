target_sources(flatter
  PRIVATE
    base.cpp
    fplll_impl.cpp
    goal.cpp
    heuristic_1.cpp
    heuristic_2.cpp
    heuristic_3.cpp
    irregular.cpp
    lagrange.cpp
    lattice_reduction.cpp
    latred_relative_sr.cpp
    schoenhage.cpp
    params.cpp
    proved_1.cpp
    proved_2.cpp
    proved_3.cpp
    recursive_generic.cpp
    sublattice_split.cpp
    sublattice_split_2.cpp
    sublattice_split_3.cpp
    threaded_3.cpp
)

# Require fplll::set_external_enumerator
find_package(FPLLL 5.1.0 REQUIRED)
target_link_libraries(flatter
  PUBLIC
    ${FPLLL_LIBRARIES}
)
target_include_directories(flatter
  PRIVATE
    ${FPLLL_INCLUDE_DIRS}
)
