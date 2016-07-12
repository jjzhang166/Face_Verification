cmake_minimum_required(VERSION 2.8.4)
project( Vgg-Joint-Bayesian )
find_package( OpenCV REQUIRED )


find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY )
add_definitions(-DENABLE_ASSERTS)

include(../../dlib-18.18/dlib/cmake)

include_directories(/usr/include/python2.7)
include_directories(/usr/include/numpy)
include_directories(src/)
add_definitions(-Xlinker -export-dynamic -Wl,-O1 -Wl,-Bsymbolic-functions -std=c++11)


add_executable(Vgg-Joint-Bayesian main.cpp src/JointBayesian.cpp src/extract_features.cpp src/FaceProcessing.cpp)


target_link_libraries( Vgg-Joint-Bayesian ${OpenCV_LIBS} )
target_link_libraries( Vgg-Joint-Bayesian dlib )
target_link_libraries( Vgg-Joint-Bayesian ${Caffe_LIBRARIES})
target_link_libraries( Vgg-Joint-Bayesian python2.7)

