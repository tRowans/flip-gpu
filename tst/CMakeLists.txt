add_executable(Tests cubic-test.cpp lattice-test.cpp)

include_directories(../src)

target_link_libraries(Tests PUBLIC relax_lib gtest_main)
