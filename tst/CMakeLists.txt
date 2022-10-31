add_executable(Tests code-test.cpp decode-test.cpp)

include_directories(../src)

target_link_libraries(Tests PRIVATE flipwrapped_lib gtest_main)
