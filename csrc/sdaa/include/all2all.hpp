#pragma once
#include "device_info.hpp"
#include "shared_memory.hpp"
#include "ib_tools.hpp"
#include <pybind11/pybind11.h>
#include <optional>
#include <string>
#include <vector>

namespace teco_ep {
struct All2allInfo {
    DeviceInfo device_info;
    std::string name;
    void* send_buffer_ptrs[8];
    void* recv_buffer_ptrs[8];
    void* send_buffer_ptrs_host;
    void* recv_buffer_ptrs_host;
    shared_memory::MemHandle send_ipc_handles[8];
    shared_memory::MemHandle recv_ipc_handles[8];
    unsigned long* send_barrier_ptrs[8];
    unsigned long* recv_barrier_ptrs[8];
    long* spa_barrier_ptrs;
    void** hidden_buffer_ptrs;
    std::vector<unsigned long> send_cnt;
    std::vector<unsigned long> recv_cnt;
    unsigned long spa_sync_cnt;
    ibv_tools::ibvResources ibv_send;
    ibv_tools::ibvResources ibv_recv;
    shared_memory::SharedMemoryAllocator shared_memory_allocator;
    bool available;

    All2allInfo(DeviceInfo& info, std::string name);
    void all2all_initialized(std::size_t cross_buffer_size, std::size_t barrier_signal_bytes,
                             std::size_t spa_barrier_bytes, std::size_t buffer_ptr_bytes);
    void all2all_released();
    pybind11::tuple get_ipc_handle() const;
    pybind11::tuple get_rdma_props() const;
    void sync_(std::size_t cross_buffer_size, std::size_t barrier_signal_bytes,
                std::size_t spa_barrier_bytes,
                const std::vector<std::optional<pybind11::tuple>>& ipc_handles,
                const std::vector<std::optional<pybind11::tuple>>& rdma_props);
    void sync_in_card();
};
static_assert(sizeof(All2allInfo) == 2472);
static_assert(offsetof(All2allInfo, send_ipc_handles) == 264);
static_assert(offsetof(All2allInfo, ibv_send) == 1616);
static_assert(offsetof(All2allInfo, available) == 2465);
} // namespace teco_ep
