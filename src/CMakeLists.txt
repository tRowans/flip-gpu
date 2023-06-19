add_executable(bpf main.cu decode.cu code.cpp) 

add_library(bpf_lib decode.cu code.cpp) 
add_library(bpfwrapped_lib decode_wrappers.cu decode.cu code.cpp)
