cmake_minimum_required(VERSION 3.16)
project(WOS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Keep all cmake/make output out of the source tree:
# configure with: cmake -S WOS -B build/cmake-wos
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_compile_options(-O3 -Wall -pedantic)
find_package(OpenMP REQUIRED)

add_executable(wos
  main.cpp
  WoSLaplace2D.cpp
  WoSPoisson2D.cpp
  helpers/geometry.cpp
)
target_link_libraries(wos PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(wos PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
