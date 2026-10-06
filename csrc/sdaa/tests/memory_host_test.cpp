#include "device_info.hpp"
#include "exception.hpp"
#include "shared_memory.hpp"
#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

namespace {
std::vector<int> calls;
std::size_t allocated_size;
void* input_ptr;
unsigned int open_flags;
shared_memory::MemHandle* observed_handle;
std::array<unsigned char, 64> ipc_bytes;
void require(bool value) { if (!value) std::abort(); }
}

extern "C" {
sdaaError_t sdaaGetDevice(int* device) { *device = 3; calls.push_back(0); return sdaaSuccess; }
sdaaError_t sdaaMallocCross(void** ptr, std::size_t size) {
    calls.push_back(1); allocated_size = size; *ptr = reinterpret_cast<void*>(0x4000); return sdaaSuccess;
}
sdaaError_t sdaaFree(void* ptr) { calls.push_back(2); input_ptr = ptr; return sdaaSuccess; }
sdaaError_t sdaaIpcGetMemHandle(sdaaIpcMemHandle_t* handle, void* ptr) {
    calls.push_back(3); input_ptr = ptr;
    require(observed_handle->size == 8193); // Original writes size before the runtime call.
    std::memcpy(handle, ipc_bytes.data(), ipc_bytes.size()); return sdaaSuccess;
}
sdaaError_t sdaaIpcOpenMemHandle(void** ptr, sdaaIpcMemHandle_t handle, unsigned int flags) {
    calls.push_back(4); open_flags = flags;
    require(std::memcmp(&handle, ipc_bytes.data(), ipc_bytes.size()) == 0);
    *ptr = reinterpret_cast<void*>(0x9000); return sdaaSuccess;
}
sdaaError_t sdaaIpcCloseMemHandle(void* ptr) { calls.push_back(5); input_ptr = ptr; return sdaaSuccess; }
const char* sdaaGetErrorString(sdaaError_t) { return "recording runtime error"; }
}

int main() {
    shared_memory::SharedMemoryAllocator allocator;
    void* ptr = nullptr;
    allocator.malloc_cross(&ptr, 8193);
    require(ptr == reinterpret_cast<void*>(0x4000) && allocated_size == 8193);
    shared_memory::MemHandle handle{};
    observed_handle = &handle;
    for (std::size_t i = 0; i < ipc_bytes.size(); ++i) ipc_bytes[i] = static_cast<unsigned char>(i * 7);
    allocator.get_mem_handle(handle, ptr, 8193);
    require(input_ptr == ptr);
    void* opened = nullptr;
    allocator.open_mem_handle(&handle, opened);
    require(opened == reinterpret_cast<void*>(0x9000) && open_flags == 0);
    allocator.close_mem_handle(opened);
    require(input_ptr == opened);
    allocator.free_cross(ptr);
    require(input_ptr == ptr && calls == std::vector<int>({1, 3, 4, 5, 2}));

    // Frozen topology cases span local card, DP partition and the 32-rank node boundary.
    teco_ep::DeviceInfo ep16(13, 16, 2, 8);
    require(ep16.device_id == 3 && ep16.node_id == 0 && ep16.node_num == 1);
    require(ep16.rank_id_local_card == 1 && ep16.rank_id_local_node == 13);
    require(ep16.card_id_local_node == 3 && ep16.num_card_per_node == 4);
    require(ep16.dp_rank == 1 && ep16.card_num_per_dp == 2 && ep16.card_id_per_dp == 1);
    require(ep16.dst_ep_with_dp == std::vector<int>({1, 3}) && ep16.local_ep_id == 3);
    teco_ep::DeviceInfo multi_node(45, 64, 4, 16);
    require(multi_node.node_id == 1 && multi_node.node_num == 2);
    require(multi_node.num_ranks_per_node == 32 && multi_node.num_card_per_node == 8);
    require(multi_node.dp_rank == 2 && multi_node.card_num_per_dp == 4);
    require(multi_node.dst_ep_with_dp == std::vector<int>({3, 7, 11, 15}));
    require(multi_node.local_ep_id == 11);
    std::ostringstream output;
    auto* previous = std::cout.rdbuf(output.rdbuf());
    ep16.printInfo(); std::cout.rdbuf(previous);
    require(output.str().find("  dst_ep_with_dp       = [ 1, 3 ]\n") != std::string::npos);
    require(output.str().find("================================\n\n") != std::string::npos);
    EPException error("failure", "original.cpp", 25);
    require(std::string(error.what()) == "failure at original.cpp:25");
}
