find_path(FPLLL_INCLUDE_DIR fplll/fplll.h)
find_library(FPLLL_LIBRARY
  NAMES fplll
)

if(FPLLL_INCLUDE_DIR)
  set(FPLLL_VERSION "")
  file(STRINGS "${FPLLL_INCLUDE_DIR}/fplll/fplll_config.h" fplll_version_str REGEX "^#define[ \t]+FPLLL_VERSION[ \t]+")
  string(REGEX REPLACE "^.*FPLLL_VERSION[ \t]+([0-9.]+)$" "\\1" FPLLL_VERSION "${fplll_version_str}")
  unset(fplll_version_str)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FPLLL
  FOUND_VAR FPLLL_FOUND
  REQUIRED_VARS FPLLL_INCLUDE_DIR FPLLL_LIBRARY
  VERSION_VAR FPLLL_VERSION
)

if(FPLLL_FOUND)
  set(FPLLL_LIBRARIES ${FPLLL_LIBRARY})
  set(FPLLL_INCLUDE_DIRS ${FPLLL_INCLUDE_DIR})
endif()
