cmake_minimum_required(VERSION 3.13)

project(flatter LANGUAGES CXX)

#flag
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_RPATH "/usr/local/lib/")

# includes
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include_directories(./include)

add_subdirectory(src)
add_subdirectory(apps)
