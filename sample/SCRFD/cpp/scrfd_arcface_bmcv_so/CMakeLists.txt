cmake_minimum_required(VERSION 3.10)
project(scrfd LANGUAGES C CXX)

# ---- 基础设置 ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # 生成位置无关代码，便于构建 .so
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# set(CMAKE_BUILD_TYPE "Debug")

# so 输出到工程根目录（也可以改为 CMAKE_LIBRARY_OUTPUT_DIRECTORY）
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
message(STATUS "cur dir: ${PROJECT_SOURCE_DIR}")

# Usage: check_and_replace_path(SDK)
function(check_and_replace_path ARG_NAME)
    if(IS_ABSOLUTE ${${ARG_NAME}})
        return()
    endif()

    set(PATH_TO_CHECK ${CMAKE_CURRENT_BINARY_DIR}/${${ARG_NAME}})

    if(EXISTS ${PATH_TO_CHECK})
        message(STATUS "Path ${PATH_TO_CHECK} exists")
        get_filename_component(ABSOLUTE_PATH ${PATH_TO_CHECK} ABSOLUTE)

        if(EXISTS ${ABSOLUTE_PATH})
            set(${ARG_NAME} ${ABSOLUTE_PATH} PARENT_SCOPE)
        else()
            message(FATAL_ERROR "Invalid path!")
        endif()
    else()
        message(FATAL_ERROR "Path ${PATH_TO_CHECK} does not exist")
    endif()
endfunction()

# ---- 目标架构 ----
if(NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie) # 可选：pcie / soc
endif()

# ---- 头文件导出（你的对外头文件目录）----
# 假设你的公共头文件都放在 include/ 下（你也可以改成 ../dependencies/include/ + 自己的 include/）
# 约定：公共头文件中尽量避免直接暴露 OpenCV/FFmpeg 类型，以免消费者也得引这些依赖
set(PUBLIC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/include")
include_directories(${PUBLIC_INCLUDE_DIR})

# ---- 源码收集 ----
# 你的对外 API 头文件建议放到 PUBLIC_INCLUDE_DIR 下，库内部实现放 src 列表
#aux_source_directory(. SRC_FILES)

set(SRC_FILES ./arcface_bm.cpp
               ./scrfd.cpp
               ./scrfd_adapter.cpp
               ./face3d.cpp
               )

aux_source_directory(../dependencies/src/ SRC_FILES)

# ---- 目标名统一为 scrfd（不同架构可用不同 OUTPUT_NAME）----
set(TGT_NAME scrfd)

if(${TARGET_ARCH} STREQUAL "pcie")
    message(STATUS "${TARGET_ARCH} mode, building shared library...")

    # -------- 查找依赖（仅用于构建 .so，消费者工程不需要再 find 这些）--------
    # FFMPEG
    set(FFMPEG_DIR /opt/sophon/sophon-ffmpeg-latest/lib/cmake)
    find_package(FFMPEG REQUIRED)
    include_directories(${FFMPEG_INCLUDE_DIRS})
    link_directories(${FFMPEG_LIB_DIRS})

    # OpenCV
    set(OpenCV_DIR /opt/sophon/sophon-opencv-latest/lib/cmake/opencv4)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIB_DIRS})

    # libsophon
    find_package(libsophon REQUIRED)
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
    link_directories(${LIBSOPHON_LIB_DIRS})

    # -------- 生成 .so --------
    add_library(${TGT_NAME} SHARED ${SRC_FILES})

    # 版本号（可选）
    set_target_properties(${TGT_NAME} PROPERTIES
        OUTPUT_NAME "scrfd_bmcv" # 生成 libscrfd_bmcv.so
        VERSION 1.0.0
        SOVERSION 1
    )

    # 仅把“你自己的公共头”导出为 PUBLIC；第三方依赖只作为 PRIVATE（不要求消费者包含）
    target_include_directories(${TGT_NAME}
        PUBLIC
        ${PUBLIC_INCLUDE_DIR}
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${FFMPEG_INCLUDE_DIRS}
        ${LIBSOPHON_INCLUDE_DIRS}
    )

    # 把依赖库都“链接进这个 .so 的依赖关系里”（消费者只需链接本 .so）
    target_link_libraries(${TGT_NAME}
        PRIVATE
        ${FFMPEG_LIBS}
        ${OpenCV_LIBS}
        ${the_libbmlib.so}
        ${the_libbmrt.so}
        ${the_libbmcv.so}
        pthread
    )

elseif(${TARGET_ARCH} STREQUAL "soc")
    message(STATUS "${TARGET_ARCH} mode, building shared library...")

    # 交叉编译工具链（如需外部 toolchain 文件，可在 cmake 命令里 -DCMAKE_TOOLCHAIN_FILE=...）
    check_and_replace_path(SDK)
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

    # 依赖库名（目标板 SDK 提供）
    set(BM_LIBS bmlib bmrt bmcv yuv)
    set(OPENCV_LIBS opencv_imgproc opencv_core opencv_highgui opencv_imgcodecs opencv_videoio opencv_calib3d)

    include_directories("${SDK}/include/")
    include_directories("${SDK}/include/opencv4")
    link_directories("${SDK}/lib/")

    message(STATUS "SDK: ${SDK}")

    add_library(${TGT_NAME} SHARED ${SRC_FILES})
    set_target_properties(${TGT_NAME} PROPERTIES
        OUTPUT_NAME "scrfd_arcface_bmcv" # 生成 libscrfd_arcface_bmcv.so
        VERSION 1.0.0
        SOVERSION 1
    )

    target_include_directories(${TGT_NAME}
        PUBLIC
        ${PUBLIC_INCLUDE_DIR}
        PRIVATE
        "${SDK}/include"
        "${SDK}/include/opencv4"
    )

    # 链到 .so 上（消费者仅需本 .so）
    target_link_libraries(${TGT_NAME}
        PRIVATE
        ${BM_LIBS}
        ${OPENCV_LIBS}
        pthread
        avcodec avformat avutil # FFMPEG（由 SDK 提供）
    )
else()
    message(FATAL_ERROR "not support mode, eg: soc/pcie, cur: ${TARGET_ARCH}")
endif()

# ---- 可选：安装规则（方便 make install 打包 headers + .so）----
# 你可以把对外头文件都放到 include/，这里示例把 ../dependencies/include 作为导出目录
install(TARGETS ${TGT_NAME}
    LIBRARY DESTINATION lib)

install(DIRECTORY ${PUBLIC_INCLUDE_DIR}/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")