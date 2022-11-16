add_executable(bpf main.cu decode.cu code.cpp) 

add_library(flip_lib decode.cu code.cpp) 
add_library(flipwrapped_lib decode_wrappers.cu decode.cu code.cpp)
