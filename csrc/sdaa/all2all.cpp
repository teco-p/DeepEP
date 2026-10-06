#include "all2all.hpp"
#include "exception.hpp"
#include "runtime_check.hpp"
#include <torch_sdaa/sdaa_extension.h>
#include <cstring>
#include <utility>

namespace teco_ep {
namespace {
constexpr const char* original_file = "/data/ci_env/slave_loongson_tecoep_py312/workspace/loongson_build_tecoep_py312/usertestdir_3/tecoep/csrc/teco_ep.cpp";
void check(bool condition, const char* expression, int line) {
    if (!condition) throw EPException(std::string("Assertion failed: ") + expression, original_file, line);
}
char* bytes(void* ptr) { return static_cast<char*>(ptr); }
}

// Original constructor inlined in Buffer at 0x1c5xx/0x1c7xx. IPC payload stays
// uninitialized until get_mem_handle/sync_; its size sentinel alone is initialized.
All2allInfo::All2allInfo(DeviceInfo& info, std::string name_)
    : device_info(info), name(name_), send_buffer_ptrs{}, recv_buffer_ptrs{},
      send_buffer_ptrs_host(nullptr), recv_buffer_ptrs_host(nullptr),
      send_barrier_ptrs{}, recv_barrier_ptrs{}, spa_barrier_ptrs(nullptr),
      hidden_buffer_ptrs(nullptr), spa_sync_cnt(0), available(false) {
    ibv_send.pending_poll_size = 0;
    ibv_recv.pending_poll_size = 0;
    for (int i = 0; i < 8; ++i) {
        send_ipc_handles[i].size = static_cast<std::size_t>(-1);
        recv_ipc_handles[i].size = static_cast<std::size_t>(-1);
    }
}

// Main ELF 0x17820. Only the first SPA rank owns this card's allocations/RDMA.
void All2allInfo::all2all_initialized(std::size_t cross_buffer_size,
    std::size_t barrier_signal_bytes, std::size_t spa_barrier_bytes, std::size_t buffer_ptr_bytes) {
    torch::sdaa::getCurrentSDAAStream(-1);
    send_cnt.resize(device_info.dp_size, 0);
    recv_cnt.resize(device_info.dp_size, 0);
    spa_sync_cnt = 0;
    if (device_info.rank_id_local_card != 0) return;
    const int card = device_info.card_id_local_node;
    const std::size_t total = cross_buffer_size + barrier_signal_bytes;
    const std::size_t send_total = total + spa_barrier_bytes + buffer_ptr_bytes;
    shared_memory_allocator.malloc_cross(&send_buffer_ptrs[card], send_total);
    shared_memory_allocator.malloc_cross(&recv_buffer_ptrs[card], total);
    ep_sdaa_check(sdaaMemset(send_buffer_ptrs[card], 0, send_total),
        "sdaaMemset((void *)send_buffer_ptrs[card_id_local_node], 0, total_size + spa_barrier_bytes + buffer_ptr_bytes)", "all2all_initialized", 148);
    ep_sdaa_check(sdaaMemset(recv_buffer_ptrs[card], 0, total),
        "sdaaMemset((void *)recv_buffer_ptrs[card_id_local_node], 0, total_size)", "all2all_initialized", 151);
    shared_memory_allocator.get_mem_handle(send_ipc_handles[card], send_buffer_ptrs[card], send_total);
    shared_memory_allocator.get_mem_handle(recv_ipc_handles[card], recv_buffer_ptrs[card], total);
    send_barrier_ptrs[card] = reinterpret_cast<unsigned long*>(bytes(send_buffer_ptrs[card]) + cross_buffer_size);
    recv_barrier_ptrs[card] = reinterpret_cast<unsigned long*>(bytes(recv_buffer_ptrs[card]) + cross_buffer_size);
    spa_barrier_ptrs = reinterpret_cast<long*>(bytes(send_buffer_ptrs[card]) + total);
    hidden_buffer_ptrs = reinterpret_cast<void**>(bytes(send_buffer_ptrs[card]) + total + spa_barrier_bytes);
    if (device_info.node_num < 2) return;
    // The original clears the complete 0x1a8-byte resource records at this point.
    std::memset(static_cast<void*>(&ibv_send), 0, sizeof(ibv_send));
    std::memset(static_cast<void*>(&ibv_recv), 0, sizeof(ibv_recv));
    ibv_send.size = total;
    ibv_recv.size = total;
    ep_sdaa_check(sdaaMemDeviceGetHostPointer(&send_buffer_ptrs_host, send_buffer_ptrs[card]),
        "sdaaMemDeviceGetHostPointer( &send_buffer_ptrs_host, send_buffer_ptrs[card_id_local_node])", "all2all_initialized", 183);
    ep_sdaa_check(sdaaMemDeviceGetHostPointer(&recv_buffer_ptrs_host, recv_buffer_ptrs[card]),
        "sdaaMemDeviceGetHostPointer( &recv_buffer_ptrs_host, recv_buffer_ptrs[card_id_local_node])", "all2all_initialized", 185);
    check(ibv_tools::resources_create(&ibv_send, send_buffer_ptrs_host, send_buffer_ptrs[card], device_info.device_id, device_info.dp_size) == 0,
        "ibv_tools::resources_create(&ibv_send, send_buffer_ptrs_host, send_buffer_ptrs[card_id_local_node], device_id, dp_size) == 0", 188);
    check(ibv_tools::resources_create(&ibv_recv, recv_buffer_ptrs_host, recv_buffer_ptrs[card], device_info.device_id, device_info.dp_size) == 0,
        "ibv_tools::resources_create(&ibv_recv, recv_buffer_ptrs_host, recv_buffer_ptrs[card_id_local_node], device_id, dp_size) == 0", 195);
    check(ibv_tools::get_con_data(&ibv_send, device_info.dp_size) == 0, "ibv_tools::get_con_data(&ibv_send, dp_size) == 0", 202);
    check(ibv_tools::get_con_data(&ibv_recv, device_info.dp_size) == 0, "ibv_tools::get_con_data(&ibv_recv, dp_size) == 0", 203);
}

// Main ELF 0x138e8. Peers close imported handles; leader frees its own allocation.
void All2allInfo::all2all_released() {
    for (int i = 0; i < device_info.num_card_per_node; ++i) {
        if (device_info.rank_id_local_card != 0 || device_info.card_id_local_node != i) {
            shared_memory_allocator.close_mem_handle(send_buffer_ptrs[i]);
            shared_memory_allocator.close_mem_handle(recv_buffer_ptrs[i]);
        }
    }
    if (device_info.rank_id_local_card != 0) return;
    if (device_info.node_num >= 2) {
        ep_sdaa_check(sdaaMemDevicePutHostPointer(send_buffer_ptrs_host), "sdaaMemDevicePutHostPointer(send_buffer_ptrs_host)", "all2all_released", 218);
        ep_sdaa_check(sdaaMemDevicePutHostPointer(recv_buffer_ptrs_host), "sdaaMemDevicePutHostPointer(recv_buffer_ptrs_host)", "all2all_released", 219);
        ibv_tools::resources_destroy(&ibv_send, device_info.dp_size);
        ibv_tools::resources_destroy(&ibv_recv, device_info.dp_size);
    }
    shared_memory_allocator.free_cross(send_buffer_ptrs[device_info.card_id_local_node]);
    shared_memory_allocator.free_cross(recv_buffer_ptrs[device_info.card_id_local_node]);
}

// Main ELF 0x15fc8 / 0x16148. Tuple payloads are bytearrays, not Python bytes.
pybind11::tuple All2allInfo::get_ipc_handle() const {
    const int card = device_info.card_id_local_node;
    return pybind11::make_tuple(
        pybind11::bytearray(reinterpret_cast<const char*>(&send_ipc_handles[card]), sizeof(shared_memory::MemHandle)),
        pybind11::bytearray(reinterpret_cast<const char*>(&recv_ipc_handles[card]), sizeof(shared_memory::MemHandle)));
}
pybind11::tuple All2allInfo::get_rdma_props() const {
    return pybind11::make_tuple(
        pybind11::bytearray(reinterpret_cast<const char*>(ibv_send.local_props.data()), ibv_send.local_props.size() * sizeof(ibv_tools::cm_con_data_t)),
        pybind11::bytearray(reinterpret_cast<const char*>(ibv_recv.local_props.data()), ibv_recv.local_props.size() * sizeof(ibv_tools::cm_con_data_t)));
}

// Main ELF 0x17cb8. Read each node's leader handle and cross-connect RDMA directions.
void All2allInfo::sync_(std::size_t cross_buffer_size, std::size_t barrier_signal_bytes,
    std::size_t spa_barrier_bytes, const std::vector<std::optional<pybind11::tuple>>& ipc_handles,
    const std::vector<std::optional<pybind11::tuple>>& rdma_props) {
    const int offset = device_info.node_id * device_info.num_ranks_per_node;
    const std::size_t total = cross_buffer_size + barrier_signal_bytes;
    for (int i = 0; i < device_info.num_ranks_per_node; i += 4) {
        check(ipc_handles[offset + i].has_value(), "ipc_handles[offset + i].has_value()", 271);
        const auto& tuple = ipc_handles[offset + i].value();
        const auto send_array = tuple[0].cast<pybind11::bytearray>();
        const auto recv_array = tuple[1].cast<pybind11::bytearray>();
        const std::string send(PyByteArray_AsString(send_array.ptr()),
            static_cast<std::size_t>(PyByteArray_Size(send_array.ptr())));
        const std::string recv(PyByteArray_AsString(recv_array.ptr()),
            static_cast<std::size_t>(PyByteArray_Size(recv_array.ptr())));
        check(send.size() == sizeof(shared_memory::MemHandle), "send_handle_str.size() == shared_memory::HANDLE_SIZE", 275);
        check(recv.size() == sizeof(shared_memory::MemHandle), "recv_handle_str.size() == shared_memory::HANDLE_SIZE", 276);
        const int card = i / 4;
        if (device_info.rank_id_local_card == 0 && device_info.card_id_local_node == card) {
            check(std::memcmp(&send_ipc_handles[card], send.data(), sizeof(shared_memory::MemHandle)) == 0,
                "std::memcmp(&send_ipc_handles[device_info.card_id_local_node], send_handle_str.data(), shared_memory::HANDLE_SIZE) == 0", 315);
            check(std::memcmp(&recv_ipc_handles[card], recv.data(), sizeof(shared_memory::MemHandle)) == 0,
                "std::memcmp(&recv_ipc_handles[device_info.card_id_local_node], recv_handle_str.data(), shared_memory::HANDLE_SIZE) == 0", 319);
        } else {
            std::memcpy(&send_ipc_handles[card], send.data(), sizeof(shared_memory::MemHandle));
            std::memcpy(&recv_ipc_handles[card], recv.data(), sizeof(shared_memory::MemHandle));
            check(static_cast<int>(send_ipc_handles[card].size) != -1, "(int)send_ipc_handles[card_id].size != -1", 289);
            check(static_cast<int>(recv_ipc_handles[card].size) != -1, "(int)recv_ipc_handles[card_id].size != -1", 290);
            shared_memory_allocator.open_mem_handle(&send_ipc_handles[card], send_buffer_ptrs[card]);
            shared_memory_allocator.open_mem_handle(&recv_ipc_handles[card], recv_buffer_ptrs[card]);
            send_barrier_ptrs[card] = reinterpret_cast<unsigned long*>(bytes(send_buffer_ptrs[card]) + cross_buffer_size);
            recv_barrier_ptrs[card] = reinterpret_cast<unsigned long*>(bytes(recv_buffer_ptrs[card]) + cross_buffer_size);
            if (device_info.card_id_local_node == card) {
                spa_barrier_ptrs = reinterpret_cast<long*>(bytes(send_buffer_ptrs[card]) + total);
                hidden_buffer_ptrs = reinterpret_cast<void**>(bytes(send_buffer_ptrs[card]) + total + spa_barrier_bytes);
            }
        }
    }
    if (device_info.node_num >= 2 && device_info.rank_id_local_card == 0) {
        ibv_send.remote_props.clear();
        ibv_recv.remote_props.clear();
        for (int i = 0; i < device_info.dp_size; ++i) {
            const int destination = device_info.dst_ep_with_dp[i] * 4;
            check(rdma_props[destination].has_value(), "rdma_props[dst_rank_id].has_value()", 332);
            const auto& tuple = rdma_props[destination].value();
            const auto send_info = pybind11::cast<pybind11::buffer>(tuple[0].cast<pybind11::bytearray>()).request();
            const auto recv_info = pybind11::cast<pybind11::buffer>(tuple[1].cast<pybind11::bytearray>()).request();
            const std::size_t expected = sizeof(ibv_tools::cm_con_data_t) * device_info.dp_size;
            check(static_cast<std::size_t>(send_info.size * send_info.itemsize) == expected,
                "(size_t)send_info.size * send_info.itemsize == (size_t)ibv_tools::CM_CON_SIZE * device_info.dp_size", 345);
            check(static_cast<std::size_t>(recv_info.size * recv_info.itemsize) == expected,
                "(size_t)recv_info.size * recv_info.itemsize == (size_t)ibv_tools::CM_CON_SIZE * device_info.dp_size", 347);
            ibv_tools::cm_con_data_t send, recv;
            std::memcpy(&send, static_cast<char*>(send_info.ptr) + device_info.dp_rank * sizeof(send), sizeof(send));
            std::memcpy(&recv, static_cast<char*>(recv_info.ptr) + device_info.dp_rank * sizeof(recv), sizeof(recv));
            ibv_send.remote_props.push_back(recv);
            ibv_recv.remote_props.push_back(send);
        }
        check(ibv_send.remote_props.size() == static_cast<std::size_t>(device_info.dp_size),
            "ibv_send.remote_props.size() == static_cast<size_t>(device_info.dp_size)", 364);
        check(ibv_recv.remote_props.size() == static_cast<std::size_t>(device_info.dp_size),
            "ibv_recv.remote_props.size() == static_cast<size_t>(device_info.dp_size)", 366);
        ibv_tools::connect_qp(&ibv_send, ibv_send.remote_props, device_info.dp_size);
        ibv_tools::connect_qp(&ibv_recv, ibv_recv.remote_props, device_info.dp_size);
    }
    ep_sdaa_check(sdaaDeviceSynchronize(), "sdaaDeviceSynchronize()", "sync_", 374);
    available = true;
}

// Main ELF 0x13a70: additive SPA flag, monotonically increasing wait generation.
void All2allInfo::sync_in_card() {
    const auto stream = torch::sdaa::getCurrentSDAAStream(-1);
    ep_sdaa_check(sdaaStreamWriteValue64(stream, &spa_barrier_ptrs[device_info.rank_id_local_card], 1, SDAA_STREAM_WRITE_VALUE_ADD),
        "sdaaStreamWriteValue64(stream, &spa_barrier_ptrs[device_info.rank_id_local_card], 1, SDAA_STREAM_WRITE_VALUE_ADD)", "sync_in_card", 382);
    ++spa_sync_cnt;
    for (int i = 0; i < 4; ++i) {
        ep_sdaa_check(sdaaStreamWaitValue64(stream, &spa_barrier_ptrs[i], spa_sync_cnt, SDAA_STREAM_WAIT_VALUE_EQ),
            "sdaaStreamWaitValue64( stream, &spa_barrier_ptrs[i], spa_sync_cnt, SDAA_STREAM_WAIT_VALUE_EQ)", "sync_in_card", 390);
    }
}
} // namespace teco_ep
