cmake_minimum_required(VERSION 3.20)
project(big_data_solo_3 C)

set(CMAKE_C_STANDARD 11)

add_executable(big_data_solo_3
        main.c
        matrix_kernels.c
        benchmark.c
)

target_compile_options(big_data_solo_3 PRIVATE -O3 -march=native -mfma)


# OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_C_FOUND)
    target_link_libraries(big_data_solo_3 PRIVATE OpenMP::OpenMP_C)
endif()

# Windows: psapi für GetProcessMemoryInfo
if (WIN32)
    target_link_libraries(big_data_solo_3 PRIVATE psapi)
endif()
