cmake_minimum_required(VERSION 3.18)
project(rvcmnn)

add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libMNN.so)

add_library(rvcmnn SHARED rvc_jni.cpp)
target_include_directories(rvcmnn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/include
                                         ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/include/MNN)
target_link_libraries(rvcmnn MNN android log)
