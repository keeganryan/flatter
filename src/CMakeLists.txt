
#flags
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

#files

#includes
find_package(OpenMP REQUIRED)
find_package(GMP REQUIRED)
find_package(LAPACK REQUIRED)
find_package(MPFR REQUIRED)

#target
set(flatter_PUBLIC_HEADER_INSTALL_DIR include/flatter)
set(flatter_PUBLIC_HEADER_DIR
  ${CMAKE_SOURCE_DIR}/include/flatter
)

add_library(flatter SHARED "")

target_include_directories(flatter
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${flatter_PUBLIC_HEADER_DIR}
)

add_subdirectory(math)
add_subdirectory(problems)
add_subdirectory(data)

target_sources(flatter
  PRIVATE
    flatter.cpp
    computation_context.cpp
    monitor.cpp
    profile.cpp
)

target_compile_options(flatter PRIVATE -Wall -Werror)

#external libraries
target_link_libraries(flatter
  PUBLIC
    ${GMP_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${MPFR_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# installation
install(TARGETS flatter
  LIBRARY DESTINATION lib
)
install(DIRECTORY ${flatter_PUBLIC_HEADER_DIR}
        DESTINATION "include"
        FILES_MATCHING
        PATTERN "*.h"
)