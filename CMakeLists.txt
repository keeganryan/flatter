cmake_minimum_required(VERSION 3.13)

project(flatter LANGUAGES CXX)

#flag
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_RPATH "/usr/local/lib/")
include_directories(/Library/Developer/CommandLineTools/usr/lib/clang/16/include)
include_directories(/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1)

set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_SYSROOT $ENV{SDKROOT})
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# includes
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include_directories(./include)

add_subdirectory(src)
add_subdirectory(apps)
