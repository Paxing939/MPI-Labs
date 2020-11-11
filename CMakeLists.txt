cmake_minimum_required(VERSION 3.17)
project(mpi_4_var)

set(CMAKE_CXX_STANDARD 14)

find_package(MPI REQUIRED)
include_directories(/usr/local/Cellar/mpich/3.3.2_1/include)

include_directories(${MPI_INCLUDE_PATH})
add_executable(mpi_4_var Lab_2.cpp)
target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES})
