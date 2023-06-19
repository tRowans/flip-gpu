add_executable(Tests code-test.cpp decode-test.cpp)

include_directories(../src)

target_link_libraries(Tests PRIVATE bpfwrapped_lib gtest_main)
