#include "shared_memory.hpp"
#include "runtime_check.hpp"

// g447683d main ELF 0x132e8..0x134e8; MemHandle layout from DWARF.
namespace shared_memory {

SharedMemoryAllocator::SharedMemoryAllocator() = default;

void SharedMemoryAllocator::malloc_cross(void** ptr, std::size_t size) {
    ep_sdaa_check(sdaaMallocCross(ptr, size), "sdaaMallocCross(ptr, size)",
                  "malloc_cross", 15);
}

void SharedMemoryAllocator::free_cross(void* ptr) {
    ep_sdaa_check(sdaaFree(ptr), "sdaaFree(ptr)", "free_cross", 18);
}

void SharedMemoryAllocator::get_mem_handle(MemHandle& mem_handle, void* ptr,
                                           std::size_t size) {
    mem_handle.size = size;
    ep_sdaa_check(sdaaIpcGetMemHandle(&mem_handle.sdaa_ipc_mem_handle, ptr),
        "sdaaIpcGetMemHandle(&mem_handle.sdaa_ipc_mem_handle, ptr)",
        "get_mem_handle", 24);
}

void SharedMemoryAllocator::open_mem_handle(MemHandle* mem_handle, void*& ptr) {
    ep_sdaa_check(sdaaIpcOpenMemHandle(&ptr, mem_handle->sdaa_ipc_mem_handle, 0),
        "sdaaIpcOpenMemHandle((void **)&ptr, mem_handle->sdaa_ipc_mem_handle, 0)",
        "open_mem_handle", 28);
}

void SharedMemoryAllocator::close_mem_handle(void* ptr) {
    ep_sdaa_check(sdaaIpcCloseMemHandle(ptr), "sdaaIpcCloseMemHandle((void *)ptr)",
                  "close_mem_handle", 33);
}

} // namespace shared_memory
