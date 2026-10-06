#pragma once

#include <cstddef>
#include <sdaa_runtime.h>
#include <sdaa_experimental_api.h>

namespace shared_memory {

struct MemHandle {
    sdaaIpcMemHandle_t sdaa_ipc_mem_handle;
    std::size_t size;
};

static_assert(sizeof(MemHandle) == 72);
static_assert(offsetof(MemHandle, size) == 64);

class SharedMemoryAllocator {
public:
    SharedMemoryAllocator();
    void malloc_cross(void** ptr, std::size_t size);
    void free_cross(void* ptr);
    void get_mem_handle(MemHandle& mem_handle, void* ptr, std::size_t size);
    void open_mem_handle(MemHandle* mem_handle, void*& ptr);
    void close_mem_handle(void* ptr);
};

} // namespace shared_memory
