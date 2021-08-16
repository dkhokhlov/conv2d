cmake_minimum_required(VERSION 3.19)
project(libconv2d)

set(CMAKE_CXX_STANDARD 11)

find_package(Python3 COMPONENTS Interpreter Development NumPy)
include_directories(${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

message("===============")
message(${Python3_INCLUDE_DIRS})
message("===============")
message(${Python3_NumPy_INCLUDE_DIRS})
message("===============")

add_library(libconv2d SHARED
            library.cpp
            wrapper.cpp
)

target_link_libraries(libconv2d ${Python3_LIBRARIES})
target_link_libraries(libconv2d ${Python3_NumPy_LIBRARIES})

target_compile_options(libconv2d  PUBLIC  -g)

set_target_properties(libconv2d PROPERTIES
                      PREFIX ""
                      OUTPUT_NAME libconv2d)

install(TARGETS libconv2d LIBRARY DESTINATION ${Python3_SITELIB})
