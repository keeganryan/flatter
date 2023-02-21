find_path(FPLLL_INCLUDE_DIR fplll/fplll.h)
find_library(FPLLL_LIBRARY
  NAMES fplll
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FPLLL
  FOUND_VAR FPLLL_FOUND
  REQUIRED_VARS
  FPLLL_INCLUDE_DIR
  FPLLL_LIBRARY
)

if(FPLLL_FOUND)
  set(FPLLL_LIBRARIES ${FPLLL_LIBRARY})
  set(FPLLL_INCLUDE_DIRS ${FPLLL_INCLUDE_DIR})
endif()
