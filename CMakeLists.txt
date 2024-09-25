cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})  # 加入這行以包含專案根目錄下的標頭檔案


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")




if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# 把 stixel_world.cpp 檔案加入可執行檔
add_executable(main main.cpp stixel_world.cpp)

# 鏈接必要的庫
target_link_libraries(main "${TORCH_LIBRARIES}")
target_link_libraries(main ${OpenCV_LIBS})

set_property(TARGET main PROPERTY CXX_STANDARD 17)


# 若是使用 Windows/MSVC，處理 DLL 檔案
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example>)
endif (MSVC)