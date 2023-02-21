cmake_minimum_required(VERSION 3.10)

project(flip-gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_COMPILER nvcc)

add_subdirectory(lib/googletest)
add_subdirectory(src)
add_subdirectory(tst)
