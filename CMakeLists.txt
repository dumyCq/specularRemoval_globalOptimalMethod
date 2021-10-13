cmake_minimum_required (VERSION 3.14.0)

project(GlobalOptimalMethodSpecularRemove)
OPTION (PROCESS_WITH_CUDA "Build VNS Standard Tools" OFF)

###############################################################################
# @qxc62 added default paths for libraries
# and then find the libraries
###############################################################################
set(OpenCV_DIR "O:/OpenCV/OpenCV-3.4.13_vs14_x64/build" CACHE PATH "OpenCV directory")
find_package(OpenCV REQUIRED)

if (PROCESS_WITH_CUDA)
	set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1" CACHE PATH "CUDA directory")
	find_package(CUDA REQUIRED)
endif()

###############################################################################
# @qxc62 added subdirectory according to if use cuda
# with options in CMAKE
###############################################################################
set(CPUTOOLS_DIR ${PROJECT_SOURCE_DIR}/CPUProcess)

if (PROCESS_WITH_CUDA)
	set(GPUCUDATOOLS_DIR ${PROJECT_SOURCE_DIR}/GPUCUDAProcess)
endif()

set(SOURCE_FILES 
	main.cpp
	${CPUTOOLS_DIR}/specularDetectCorrect.cpp
	${CPUTOOLS_DIR}/convexWeightsEstimation.cpp
)

set(HEADER_FILES
	${CPUTOOLS_DIR}/specularDetectCorrect.h
	${CPUTOOLS_DIR}/convexWeightsEstimation.h
)


###############################################################################
# @qxc62 if use cuda
# call nvcc to compile
###############################################################################
if (PROCESS_WITH_CUDA)
	set(CMAKE_NVCC_FLAGS "CMAKE_NVCC_FLAGS -std=c++14")
	include_directories(${OpenCV_INCLUDE_DIRS} "${CUDA_INCLUDE_DIRS}" ${HEADER_FILES})
	cuda_add_executable(GlobalOptimalMethodSpecularRemove ${SOURCE_FILES} ${HEADER_FILES})
	target_link_libraries(GlobalOptimalMethodSpecularRemove ${OpenCV_LIBS} ${CUDA_RUNTIME_LIBRARY})
else()
	include_directories(${OpenCV_INCLUDE_DIRS} "${CUDA_INCLUDE_DIRS}" ${HEADER_FILES})
	add_executable(GlobalOptimalMethodSpecularRemove ${SOURCE_FILES} ${HEADER_FILES})
	target_link_libraries(GlobalOptimalMethodSpecularRemove ${OpenCV_LIBS})
endif()





#

