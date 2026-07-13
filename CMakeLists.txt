cmake_minimum_required(VERSION 3.20)
project(jacobi_schwarz_mpi C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")

find_package(MPI REQUIRED)

set(SRC_COMUNI poisson.c thomas.c jacobi.c schwarz.c)
add_executable(solver main.c ${SRC_COMUNI})
target_link_libraries(solver PRIVATE MPI::MPI_C m)  # m = libreria matematica (-lm)
