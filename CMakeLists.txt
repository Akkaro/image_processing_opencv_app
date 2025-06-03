cmake_minimum_required(VERSION 3.10)
project(PhotoEditor)

set(CMAKE_CXX_STANDARD 17)

# Try to find OpenCV
find_package(OpenCV REQUIRED)

# If OpenCV is not found automatically, you might need to set the path manually
# Uncomment and modify the path below if needed:
# set(OpenCV_DIR "C:/path/to/your/opencv/build")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(photo_editor photo_editor.cpp)

target_link_libraries(photo_editor ${OpenCV_LIBS})

# For Windows - copy OpenCV DLLs to output directory
if(WIN32)
    set_target_properties(photo_editor PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()