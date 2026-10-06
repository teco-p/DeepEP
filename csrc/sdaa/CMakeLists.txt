cmake_minimum_required(VERSION 3.20)
project(tecoep_native_recovery LANGUAGES CXX)

set(SDAA_INCLUDE_DIR "" CACHE PATH "Pinned SDK directory containing sdaa_runtime.h")
if(NOT EXISTS "${SDAA_INCLUDE_DIR}/sdaa_runtime.h")
    message(FATAL_ERROR "Set SDAA_INCLUDE_DIR to the fixed SDK include directory")
endif()

# Object targets allow the affected compile gates to reuse completed objects.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(tecoep_dispatch_host OBJECT dispatch_host.cpp)
target_compile_features(tecoep_dispatch_host PUBLIC cxx_std_17)
target_include_directories(tecoep_dispatch_host PUBLIC include "${SDAA_INCLUDE_DIR}")
target_compile_options(tecoep_dispatch_host PRIVATE -Wall -Wextra -Werror)

add_library(tecoep_memory_host OBJECT shared_memory.cpp device_info.cpp)
target_compile_features(tecoep_memory_host PUBLIC cxx_std_17)
target_include_directories(tecoep_memory_host PUBLIC include "${SDAA_INCLUDE_DIR}")
target_compile_options(tecoep_memory_host PRIVATE -Wall -Wextra -Werror)

add_library(tecoep_combine_host OBJECT combine_host.cpp)
target_compile_features(tecoep_combine_host PUBLIC cxx_std_17)
target_include_directories(tecoep_combine_host PUBLIC include "${SDAA_INCLUDE_DIR}")
target_compile_options(tecoep_combine_host PRIVATE -Wall -Wextra -Werror)

add_library(tecoep_epoch_host OBJECT epoch_host.cpp)
target_compile_features(tecoep_epoch_host PUBLIC cxx_std_17)
target_include_directories(tecoep_epoch_host PUBLIC include "${SDAA_INCLUDE_DIR}")
target_compile_options(tecoep_epoch_host PRIVATE -Wall -Wextra -Werror)

set(SDAA_COMPILER "" CACHE FILEPATH "Pinned tecocc compiler")
if(SDAA_COMPILER)
    set(device_objects)
    foreach(device_source dispatch_device gather_device combine_device epoch_device routing_device)
        set(device_dependencies "${device_source}.scpp")
        if(NOT device_source STREQUAL "gather_device")
            list(APPEND device_dependencies include/device_sync.hpp)
        endif()
        if(device_source STREQUAL "routing_device")
            list(APPEND device_dependencies include/expanded_layout.hpp)
        endif()
        add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${device_source}.o"
            COMMAND "${SDAA_COMPILER}" -I "${SDAA_INCLUDE_DIR}"
                -U__CUDA_ARCH__ -DUSE_EXPERIMENTAL_SDAA_API -flto -fPIC
                -std=c++20 -D_GLIBCXX_USE_CXX11_ABI=1 -c
                "${CMAKE_CURRENT_SOURCE_DIR}/${device_source}.scpp"
                -o "${CMAKE_CURRENT_BINARY_DIR}/${device_source}.o"
            DEPENDS ${device_dependencies} VERBATIM)
        add_custom_target("tecoep_${device_source}"
            DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${device_source}.o")
        list(APPEND device_objects "${CMAKE_CURRENT_BINARY_DIR}/${device_source}.o")
    endforeach()
    # tecocc generates the global stubs and fatbin registration from the
    # explicit SCPP template instantiations. Reuse its original link driver.
    # Native ELF host objects go straight to the final host linker; passing
    # them as offload inputs makes the device llvm-link parse ELF as bitcode.
    set(device_library "${CMAKE_CURRENT_BINARY_DIR}/libdeep_ep_sdaa_kernels.so")
    add_custom_command(OUTPUT "${device_library}"
        COMMAND "${SDAA_COMPILER}"
            ${device_objects}
            -Xlinker $<TARGET_OBJECTS:tecoep_dispatch_host>
            -Xlinker $<TARGET_OBJECTS:tecoep_combine_host>
            -Xlinker $<TARGET_OBJECTS:tecoep_epoch_host>
            -o "${device_library}" -flto -fPIC -shared --sdaa-link
            -fuse-ld=lld -lm
        DEPENDS tecoep_dispatch_host tecoep_combine_host tecoep_epoch_host ${device_objects}
        COMMAND_EXPAND_LISTS VERBATIM)
    add_custom_target(tecoep_device_library DEPENDS "${device_library}")
endif()

set(TORCH_SDAA_INCLUDE_DIR "" CACHE PATH "Pinned TorchSDAA include directory")
if(TORCH_SDAA_INCLUDE_DIR)
    find_package(Torch REQUIRED)
    add_library(tecoep_torch_ops OBJECT ops.cpp)
    target_compile_features(tecoep_torch_ops PUBLIC cxx_std_17)
    target_include_directories(tecoep_torch_ops PUBLIC include "${SDAA_INCLUDE_DIR}" "${TORCH_SDAA_INCLUDE_DIR}")
    target_link_libraries(tecoep_torch_ops PRIVATE ${TORCH_LIBRARIES})
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
    find_path(IBVERBS_INCLUDE_DIR infiniband/verbs.h REQUIRED)
    add_library(tecoep_transport_host OBJECT config.cpp ib_tools.cpp ib_comm.cpp
        all2all.cpp buffer_lifecycle.cpp buffer_kernels.cpp buffer_communication.cpp bindings.cpp routing_bindings.cpp)
    target_compile_features(tecoep_transport_host PUBLIC cxx_std_17)
    target_include_directories(tecoep_transport_host PUBLIC include "${SDAA_INCLUDE_DIR}"
        "${TORCH_SDAA_INCLUDE_DIR}" "${IBVERBS_INCLUDE_DIR}")
    target_link_libraries(tecoep_transport_host PRIVATE ${TORCH_LIBRARIES} Python3::Module)

    if(SDAA_COMPILER)
        get_filename_component(sdaa_root "${SDAA_INCLUDE_DIR}" DIRECTORY)
        get_filename_component(torch_sdaa_root "${TORCH_SDAA_INCLUDE_DIR}" DIRECTORY)
        find_library(SDAART_LIBRARY sdaart HINTS "${sdaa_root}/lib64" REQUIRED)
        find_library(IBVERBS_LIBRARY ibverbs REQUIRED)
        find_library(TORCH_PYTHON_LIBRARY torch_python
            HINTS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
        find_library(C10_SDAA_LIBRARY c10_sdaa
            HINTS "${torch_sdaa_root}/lib" REQUIRED)
        find_library(TORCH_SDAA_LIBRARY torch_sdaa
            HINTS "${torch_sdaa_root}/lib" REQUIRED)
        Python3_add_library(_sdaa MODULE WITH_SOABI
            $<TARGET_OBJECTS:tecoep_memory_host>
            $<TARGET_OBJECTS:tecoep_torch_ops>
            $<TARGET_OBJECTS:tecoep_transport_host>)
        set_target_properties(_sdaa PROPERTIES LINKER_LANGUAGE CXX BUILD_RPATH "$ORIGIN"
            INSTALL_RPATH "$ORIGIN" LINK_DEPENDS "${device_library}")
        # The tecocc auxiliary library has no SONAME. Link by its library name
        # so DT_NEEDED and $ORIGIN remain valid after installation.
        target_link_directories(_sdaa PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
        target_link_libraries(_sdaa PRIVATE deep_ep_sdaa_kernels
            ${TORCH_LIBRARIES} "${TORCH_PYTHON_LIBRARY}"
            "${C10_SDAA_LIBRARY}" "${TORCH_SDAA_LIBRARY}"
            "${SDAART_LIBRARY}" "${IBVERBS_LIBRARY}")
        add_dependencies(_sdaa tecoep_device_library)
        install(TARGETS _sdaa LIBRARY DESTINATION .)
        install(FILES "${device_library}" DESTINATION .)
    endif()
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(dispatch_host_test tests/dispatch_host_test.cpp)
    target_link_libraries(dispatch_host_test PRIVATE tecoep_dispatch_host)
    add_test(NAME dispatch_host_contract COMMAND dispatch_host_test)
    add_executable(memory_host_test tests/memory_host_test.cpp)
    target_link_libraries(memory_host_test PRIVATE tecoep_memory_host)
    add_test(NAME memory_host_contract COMMAND memory_host_test)
    add_executable(combine_host_test tests/combine_host_test.cpp)
    target_link_libraries(combine_host_test PRIVATE tecoep_combine_host)
    add_test(NAME combine_host_contract COMMAND combine_host_test)
endif()
