# SPDX-FileCopyrightText: 2025 Ervin Tasnadi <etasnadi@protonmail.com>
# SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.10.2)

set(CMAKE_BUILD_TYPE Debug)

project(spda_flash)

find_package(Vulkan REQUIRED)

include_directories(${Vulkan_INCLUDE_DIR})

add_executable(test_sdpa_flash src/sdpa_flash.cpp )
add_library(sdpa_flash SHARED src/sdpa_flash.cpp)

target_link_libraries(test_sdpa_flash ${Vulkan_LIBRARY})
target_link_libraries(sdpa_flash ${Vulkan_LIBRARY})

target_compile_features(test_sdpa_flash PRIVATE cxx_std_23)
target_compile_features(sdpa_flash PRIVATE cxx_std_23)

#target_compile_definitions(test_sdpa_flash PRIVATE DEBUG_MODE)
#target_compile_definitions(sdpa_flash PRIVATE DEBUG_MODE)

# Build the shaders, and add as depdendencies to the main application

set(SHADER_DIR shaders)

#file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_DIR})

set(GLSLANG_ARGS 
    "--target-env spirv1.3 -DA_BITS=16 -DA_TYPE=float16_t -DS_BITS=16 -DS_TYPE=float16_t -DC_BITS=16 -DC_TYPE=float16_t"
    "--target-env spirv1.3 -DA_BITS=16 -DA_TYPE=float16_t -DS_BITS=16 -DS_TYPE=float16_t -DC_BITS=32 -DC_TYPE=float"
    "--target-env spirv1.3 -DA_BITS=16 -DA_TYPE=float16_t -DS_BITS=32 -DS_TYPE=float -DC_BITS=16 -DC_TYPE=float16_t"    
    "--target-env spirv1.3 -DA_BITS=16 -DA_TYPE=float16_t -DS_BITS=32 -DS_TYPE=float -DC_BITS=32 -DC_TYPE=float"    
)

set(SHADER_INPUT_FILES
    "sdpa_flash.comp"
    "sdpa_flash.comp"
    "sdpa_flash.comp"
    "sdpa_flash.comp"        
)

set(SHADER_OUTPUT_FILES
    "sdpa_flash-sfp16-cfp16.spv"
    "sdpa_flash-sfp16-cfp32.spv"    
    "sdpa_flash-sfp32-cfp16.spv"
    "sdpa_flash-sfp32-cfp32.spv"
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_DIR})
add_custom_target(create_shader_dir ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_DIR})

foreach(args input output IN ZIP_LISTS GLSLANG_ARGS SHADER_INPUT_FILES SHADER_OUTPUT_FILES)
    # TODO: it is just an instant solution, no idea how to achieve this simple thing otherwise.
    string(REPLACE " " ";" ARGS_EXPLODED ${args})
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_DIR}/${output}
        COMMAND glslangValidator ${ARGS_EXPLODED} -V ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_DIR}/${input} -o ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_DIR}/${output}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_DIR}/${input} ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_DIR})
    add_custom_target(${output} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_DIR}/${output})
    
    add_dependencies(sdpa_flash ${output})

endforeach()
