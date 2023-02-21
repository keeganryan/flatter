find_path(GMP_INCLUDE_DIR gmp.h)
find_library(GMP_LIBRARY
  NAMES gmp
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMP
  FOUND_VAR GMP_FOUND
  REQUIRED_VARS
  GMP_INCLUDE_DIR
  GMP_LIBRARY
)

if(GMP_FOUND)
  set(GMP_LIBRARIES ${GMP_LIBRARY})
  set(GMP_INCLUDE_DIRS ${GMP_INCLUDE_DIR})
endif()
