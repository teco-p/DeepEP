#include "buffer.hpp"
#include "exception.hpp"
#include "runtime_check.hpp"
#include <cstdlib>

void getEnvFromFile();

namespace teco_ep {
namespace {
constexpr const char* original_file = "/data/ci_env/slave_loongson_tecoep_py312/workspace/loongson_build_tecoep_py312/usertestdir_3/tecoep/csrc/teco_ep.cpp";
void check(bool condition, const char* expression, int line) {
    if (!condition) throw EPException(std::string("Assertion failed: ") + expression, original_file, line);
}
}

// Main ELF 0x1c520. The original capacity check only rejects zero, despite its text.
Buffer::Buffer(int rank_, int num_ranks_, int dp_size_, int ep_size_,
    std::size_t capacity, bool low_latency)
    : rank(rank_), num_ranks(num_ranks_), dp_size(dp_size_), ep_size(ep_size_),
      cross_buffer_size(capacity), low_latency_mode(low_latency),
      device_info(rank_, num_ranks_, dp_size_, ep_size_), dispatch(device_info, "dispatch"),
      combine(device_info, "combine") {
    getEnvFromFile();
    dp_rank = device_info.dp_rank;
    check(dp_size > 0 && dp_size <= 32, "DP size must be in [1, 32]", 417);
    check(!low_latency_mode, "SDAA Graph backend does not support low latency", 417);
    for (int i = 0; i < dp_size; ++i) {
        check(is_same_node(device_info.local_ep_id, device_info.dst_ep_with_dp[i]),
            "SDAA Graph backend requires single-node peers", 417);
    }
    if (num_ranks % 4 != 0) throw EPException(
        "(num_ranks % MAX_SPA_NUM_PER_CARD) == 0failed, World_size must be a mutiple of 4 (SPA NUM).", original_file, 417);
    if (capacity == 0) throw EPException(
        "cross_buffer_size > 0failed, cross_buffer_siez must be greater than 64 MiB.", original_file, 419);
    spa_barrier_bytes = 32;
    barrier_signal_bytes = static_cast<std::size_t>(dp_size) * 8;
    buffer_ptr_bytes = 32;
    const auto cpu_int = at::TensorOptions().dtype(at::kInt).device(at::kCPU);
    dispatch_num_tokens_cpu = at::empty({ep_size}, cpu_int);
    combine_num_tokens_cpu = at::empty({dp_size + 1}, cpu_int);
    dispatch.all2all_initialized(capacity, barrier_signal_bytes, spa_barrier_bytes, buffer_ptr_bytes);
    combine.all2all_initialized(capacity, barrier_signal_bytes, spa_barrier_bytes, buffer_ptr_bytes);
    intra_node_dst = static_cast<int*>(std::malloc(static_cast<std::size_t>(dp_size) * 4));
    inter_node_dst = static_cast<int*>(std::malloc(static_cast<std::size_t>(dp_size) * 4));
    for (int i = 0; i < dp_size; ++i) {
        if (device_info.dp_rank == i) { intra_node_dst[i] = -1; inter_node_dst[i] = -1; }
        else if (is_same_node(device_info.local_ep_id, device_info.dst_ep_with_dp[i])) {
            intra_node_dst[i] = device_info.dst_ep_with_dp[i]; inter_node_dst[i] = -1;
        } else { intra_node_dst[i] = -1; inter_node_dst[i] = device_info.dst_ep_with_dp[i]; }
    }
    int ordinal;
    ep_sdaa_check(sdaaGetDevice(&ordinal), "sdaaGetDevice", "Buffer", 445);
    const auto epoch_options = at::TensorOptions().dtype(at::kLong)
        .device(at::Device(at::kPrivateUse1, ordinal));
    dispatch_expected_epoch_ = at::zeros({dp_size}, epoch_options);
    combine_expected_epoch_ = at::zeros({dp_size}, epoch_options);
    epoch_status_ = at::zeros({4}, epoch_options);
    ep_sdaa_check(sdaaMalloc(&workspace_, capacity), "sdaaMalloc(workspace)", "Buffer", 445);
    ep_sdaa_check(sdaaMalloc(reinterpret_cast<void**>(&barrier_flag_), static_cast<std::size_t>(dp_size) * 32),
        "sdaaMalloc(barrier)", "Buffer", 446);
    ep_sdaa_check(sdaaMemset(barrier_flag_, 0, static_cast<std::size_t>(dp_size) * 32),
        "sdaaMemset(barrier)", "Buffer", 447);
}

// Main ELF 0x1b2d8; C++ members release after the explicit native resources.
Buffer::~Buffer() {
    std::free(intra_node_dst);
    std::free(inter_node_dst);
    dispatch.all2all_released();
    combine.all2all_released();
    sdaaFree(workspace_);
    sdaaFree(barrier_flag_);
}
int Buffer::get_num_nodes() const { return device_info.node_num; }
bool Buffer::is_available() const { return dispatch.available && combine.available; }
bool Buffer::is_same_node(int left, int right) {
    return left / device_info.num_card_per_node == right / device_info.num_card_per_node;
}
pybind11::tuple Buffer::get_dispatch_ipc_handle() const { return dispatch.get_ipc_handle(); }
pybind11::tuple Buffer::get_combine_ipc_handle() const { return combine.get_ipc_handle(); }
pybind11::tuple Buffer::get_dispatch_props() const { return dispatch.get_rdma_props(); }
pybind11::tuple Buffer::get_combine_props() const { return combine.get_rdma_props(); }

// Main ELF 0x190e8.
void Buffer::sync(const std::vector<std::optional<pybind11::tuple>>& dispatch_ipc_handles,
    const std::vector<std::optional<pybind11::tuple>>& dispatch_rdma_props,
    const std::vector<std::optional<pybind11::tuple>>& combine_ipc_handles,
    const std::vector<std::optional<pybind11::tuple>>& combine_rdma_props) {
    check(static_cast<int>(dispatch_ipc_handles.size()) == num_ranks, "(int)dispatch_ipc_handles.size() == num_ranks", 497);
    check(static_cast<int>(dispatch_rdma_props.size()) == num_ranks, "(int)dispatch_rdma_props.size() == num_ranks", 498);
    check(static_cast<int>(combine_ipc_handles.size()) == num_ranks, "(int)combine_ipc_handles.size() == num_ranks", 499);
    check(static_cast<int>(combine_rdma_props.size()) == num_ranks, "(int)combine_rdma_props.size() == num_ranks", 500);
    dispatch.sync_(cross_buffer_size, barrier_signal_bytes, spa_barrier_bytes, dispatch_ipc_handles, dispatch_rdma_props);
    combine.sync_(cross_buffer_size, barrier_signal_bytes, spa_barrier_bytes, combine_ipc_handles, combine_rdma_props);
}

// Main ELF 0x146e8 / 0x14930. Unknown names are no-ops; only put is leader-only.
void Buffer::cross_buffer_put(const at::Tensor& tensor, const std::string& name) {
    if (device_info.rank_id_local_card != 0) return;
    const int card = device_info.card_id_local_node;
    if (name == "dispatch") {
        ep_sdaa_check(sdaaMemcpy(dispatch.send_buffer_ptrs[card], tensor.const_data_ptr(),
            tensor.numel() * tensor.element_size(), sdaaMemcpyDeviceToDevice),
            "sdaaMemcpy(dispatch.send_buffer_ptrs[device_info.card_id_local_node], tensor.const_data_ptr(), tensor.numel() * tensor.element_size(), sdaaMemcpyDeviceToDevice)", "cross_buffer_put", 1119);
    } else if (name == "combine") {
        ep_sdaa_check(sdaaMemcpy(combine.send_buffer_ptrs[card], tensor.const_data_ptr(),
            tensor.numel() * tensor.element_size(), sdaaMemcpyDeviceToDevice),
            "sdaaMemcpy(combine.send_buffer_ptrs[device_info.card_id_local_node], tensor.const_data_ptr(), tensor.numel() * tensor.element_size(), sdaaMemcpyDeviceToDevice)", "cross_buffer_put", 1125);
    }
}
void Buffer::cross_buffer_get(at::Tensor& tensor, const std::string& name) {
    const int card = device_info.card_id_local_node;
    if (name == "dispatch") {
        ep_sdaa_check(sdaaMemcpy(tensor.data_ptr(), dispatch.recv_buffer_ptrs[card],
            tensor.numel() * tensor.element_size(), sdaaMemcpyDeviceToDevice),
            "sdaaMemcpy(tensor.data_ptr(), dispatch.recv_buffer_ptrs[device_info.card_id_local_node], tensor.numel() * tensor.element_size(), sdaaMemcpyDeviceToDevice)", "cross_buffer_get", 1136);
    } else if (name == "combine") {
        ep_sdaa_check(sdaaMemcpy(tensor.data_ptr(), combine.recv_buffer_ptrs[card],
            tensor.numel() * tensor.element_size(), sdaaMemcpyDeviceToDevice),
            "sdaaMemcpy(tensor.data_ptr(), combine.recv_buffer_ptrs[device_info.card_id_local_node], tensor.numel() * tensor.element_size(), sdaaMemcpyDeviceToDevice)", "cross_buffer_get", 1142);
    }
}
} // namespace teco_ep
