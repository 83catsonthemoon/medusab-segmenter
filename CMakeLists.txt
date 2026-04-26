cmake_minimum_required(VERSION 3.24)

set(MEDUSAB_SEGMENTER_VERSION "0.0.0" CACHE STRING "Application version")

project(medusab-segmenter VERSION ${MEDUSAB_SEGMENTER_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)

find_package(Qt6 REQUIRED COMPONENTS Widgets Network)

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

if (MSVC)
    # warning level 4
    add_compile_options(/W4 /Qpar /GF /favor:blend /arch:AVX2)
else()
    # additional warnings
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -flto -mavx2 -mfma)
endif()

qt_standard_project_setup()

qt_add_executable(medusab-segmenter
    src/main.cpp
    src/mainwindow.cpp
    src/mainwindow.h
    src/segmentationsession.cpp
    src/segmentationsession.h
)

target_link_libraries(medusab-segmenter PRIVATE Qt6::Widgets Qt6::Network onnxruntime)
target_include_directories(medusab-segmenter PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/modules/core/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/modules/imgproc/include"
    "${CMAKE_CURRENT_BINARY_DIR}"
)
target_link_libraries(medusab-segmenter PRIVATE opencv_core opencv_imgproc)

set_target_properties(opencv_core opencv_imgproc PROPERTIES
    INSTALL_RPATH "$ORIGIN"
)

set_target_properties(medusab-segmenter PROPERTIES
    WIN32_EXECUTABLE ON
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
)

install(TARGETS medusab-segmenter opencv_core opencv_imgproc
    BUNDLE DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(WIN32)
    file(GLOB ONNXRUNTIME_RUNTIME_FILES "${ONNXRUNTIME_ROOT}/lib/*.dll")
    install(FILES ${ONNXRUNTIME_RUNTIME_FILES} DESTINATION ${CMAKE_INSTALL_BINDIR})
else()
    file(GLOB ONNXRUNTIME_RUNTIME_FILES "${ONNXRUNTIME_ROOT}/lib/libonnxruntime*.so*")
    install(FILES ${ONNXRUNTIME_RUNTIME_FILES} DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/medusab-segmenter)

if(UNIX AND NOT APPLE)
    install(FILES packaging/linux/medusab-segmenter.desktop
        DESTINATION ${CMAKE_INSTALL_DATADIR}/applications
    )
    install(FILES assets/icon.png
        DESTINATION ${CMAKE_INSTALL_DATADIR}/icons/hicolor/512x512/apps
        RENAME medusab-segmenter.png
    )
endif()

if(WIN32)
    qt_generate_deploy_app_script(
        TARGET medusab-segmenter
        FILENAME_VARIABLE qt_deploy_script
        NO_UNSUPPORTED_PLATFORM_ERROR
    )
    install(SCRIPT ${qt_deploy_script})
endif()

set(CPACK_PACKAGE_NAME "medusab-segmenter")
set(CPACK_PACKAGE_VENDOR "83catsonthemoon")
set(CPACK_PACKAGE_CONTACT "83catsonthemoon@users.noreply.github.com")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Desktop image segmentation app powered by Qt and ONNX Runtime")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_INSTALL_DIRECTORY "Medusab Segmenter")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")
    set(CPACK_NSIS_DISPLAY_NAME "Medusab Segmenter")
    set(CPACK_NSIS_PACKAGE_NAME "Medusab Segmenter")
else()
    set(CPACK_GENERATOR "TGZ;DEB")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "83catsonthemoon")
    set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
    set(CPACK_DEBIAN_PACKAGE_SECTION graphics)
    set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
endif()

include(CPack)
