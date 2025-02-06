cmake_minimum_required(VERSION 3.10)
project(LaserDetectionDemo)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./dist")

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

add_definitions(-DOPENCV_DISABLE_EIGEN_TENSOR_SUPPORT)

set(SOURCE ./src/main.cpp)
add_executable(app ${SOURCE})

target_link_libraries(app ${OpenCV_LIBS})
target_link_libraries(app Eigen3::Eigen)
