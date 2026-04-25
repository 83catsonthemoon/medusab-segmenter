cmake_minimum_required(VERSION 3.24)

project(medusab-segmenter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets)

set(BUILD_LIST core,imgproc CACHE STRING "OpenCV modules to build" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "Disable OpenCV tests" FORCE)
set(BUILD_PERF_TESTS OFF CACHE BOOL "Disable OpenCV performance tests" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable OpenCV examples" FORCE)
set(BUILD_opencv_apps OFF CACHE BOOL "Disable OpenCV apps" FORCE)
set(BUILD_opencv_gapi OFF CACHE BOOL "Disable OpenCV G-API" FORCE)
set(BUILD_opencv_python_bindings_generator OFF CACHE BOOL "Disable OpenCV Python bindings generator" FORCE)
set(BUILD_opencv_java_bindings_generator OFF CACHE BOOL "Disable OpenCV Java bindings generator" FORCE)
set(WITH_IPP OFF CACHE BOOL "Disable OpenCV IPP download" FORCE)
set(WITH_JPEG OFF CACHE BOOL "Disable OpenCV JPEG support" FORCE)
set(BUILD_JPEG OFF CACHE BOOL "Disable bundled JPEG" FORCE)
set(WITH_OPENJPEG OFF CACHE BOOL "Disable OpenCV JPEG 2000 support" FORCE)
set(BUILD_OPENJPEG OFF CACHE BOOL "Disable bundled OpenJPEG" FORCE)
set(WITH_PNG OFF CACHE BOOL "Disable OpenCV PNG support" FORCE)
set(BUILD_PNG OFF CACHE BOOL "Disable bundled PNG" FORCE)
set(WITH_TIFF OFF CACHE BOOL "Disable OpenCV TIFF support" FORCE)
set(BUILD_TIFF OFF CACHE BOOL "Disable bundled TIFF" FORCE)
set(WITH_WEBP OFF CACHE BOOL "Disable OpenCV WebP support" FORCE)
set(BUILD_WEBP OFF CACHE BOOL "Disable bundled WebP" FORCE)
set(WITH_OPENEXR OFF CACHE BOOL "Disable OpenCV OpenEXR support" FORCE)
set(BUILD_OPENEXR OFF CACHE BOOL "Disable bundled OpenEXR" FORCE)
add_subdirectory(opencv EXCLUDE_FROM_ALL)

set(ONNXRUNTIME_ROOT "" CACHE PATH "./onnxruntime-linux-x64-gpu-1.25.0/")

add_library(onnxruntime SHARED IMPORTED)

if(WIN32)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_IMPLIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
        IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
    )
else()
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
    )
endif()

qt_standard_project_setup()

qt_add_executable(medusab-segmenter
    src/main.cpp
    src/mainwindow.cpp
    src/mainwindow.h
    src/segmentationsession.cpp
    src/segmentationsession.h
)

target_link_libraries(medusab-segmenter PRIVATE Qt6::Widgets onnxruntime)
target_include_directories(medusab-segmenter PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/modules/core/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/modules/imgproc/include"
    "${CMAKE_CURRENT_BINARY_DIR}"
)
target_link_libraries(medusab-segmenter PRIVATE opencv_core opencv_imgproc)

set_target_properties(medusab-segmenter PROPERTIES WIN32_EXECUTABLE ON)
