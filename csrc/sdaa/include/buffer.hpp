#pragma once
#include "all2all.hpp"
#include <ATen/ATen.h>
#include <cstdint>

namespace teco_ep {
struct Buffer {
private:
    int rank, num_ranks, dp_size, dp_rank, ep_size;
    std::size_t cross_buffer_size, barrier_signal_bytes, spa_barrier_bytes, buffer_ptr_bytes;
    bool low_latency_mode;
    DeviceInfo device_info;
    at::Tensor dispatch_num_tokens_cpu;
    at::Tensor combine_num_tokens_cpu;
    int* intra_node_dst;
    int* inter_node_dst;
    All2allInfo dispatch;
    All2allInfo combine;
    void* workspace_;
    unsigned long* barrier_flag_;
    at::Tensor dispatch_expected_epoch_;
    at::Tensor combine_expected_epoch_;
    at::Tensor epoch_status_;
    std::int64_t dispatch_token_bound_ = -1;
public:
    Buffer(int rank, int num_ranks, int dp_size, int ep_size, std::size_t cross_buffer_size, bool low_latency_mode);
    ~Buffer();
    int get_num_nodes() const;
    bool is_available() const;
    // All DP sources must configure the same capacity before any capture/call.
    void set_transport_token_capacity(std::int64_t capacity) {
        TORCH_CHECK(capacity >= 0 && (dispatch_token_bound_ < 0 || dispatch_token_bound_ == capacity),
            "transport token capacity must be nonnegative and immutable");
        dispatch_token_bound_ = capacity;
    }
    const at::Tensor& get_epoch_status() const { return epoch_status_; }
    pybind11::tuple get_dispatch_ipc_handle() const;
    pybind11::tuple get_combine_ipc_handle() const;
    pybind11::tuple get_dispatch_props() const;
    pybind11::tuple get_combine_props() const;
    void sync(const std::vector<std::optional<pybind11::tuple>>& dispatch_ipc_handles,
               const std::vector<std::optional<pybind11::tuple>>& dispatch_rdma_props,
               const std::vector<std::optional<pybind11::tuple>>& combine_ipc_handles,
               const std::vector<std::optional<pybind11::tuple>>& combine_rdma_props);
    bool is_same_node(int left, int right);
    void cross_buffer_put(const at::Tensor& input, const std::string& name);
    void cross_buffer_get(at::Tensor& output, const std::string& name);
    void get_dispatch_layout(const at::Tensor& topk_idx, const at::Tensor& topk_weight,
        const at::Tensor& hidden_states, int local_expert_num, int global_expert_num,
        int ep_size, int dp_size, at::Tensor& is_token_in_card, at::Tensor& num_tokens_per_card);
    void gather_dispatch_data(long dp_rank, long dp_size, const at::Tensor& recv_num_tokens_per_dp,
        const at::Tensor& recv_topk_idx, const at::Tensor& recv_topk_weight, const at::Tensor& recv_hidden_states);
    void get_combine_layout(const at::Tensor& expert_hidden, const at::Tensor& recv_num_tokens_per_dp,
        int ep_size, int dp_size);
    void gather_combine_data(long dp_rank, long dp_size, long ep_size,
        const at::Tensor& is_token_in_card, const at::Tensor& num_tokens_per_card, at::Tensor& out_hidden_states);
    void dispatch_data(const at::Tensor& topk_idx, const at::Tensor& topk_weight,
        const at::Tensor& hidden_states, int ep_size, int dp_size, const at::Tensor& num_tokens_per_card);
    void combine_data(const at::Tensor& hidden_states, int ep_size, int dp_size, const at::Tensor& recv_num_tokens_per_dp);
    void dispatch_data_lowlatency(const at::Tensor& topk_idx, const at::Tensor& topk_weight,
        const at::Tensor& hidden_states, int ep_size, int dp_size, const at::Tensor& num_tokens_per_card);
    void combine_data_lowlatency(const at::Tensor& hidden_states, int ep_size, int dp_size, const at::Tensor& recv_num_tokens_per_dp);
    void debug_dump_combine_headers(at::Tensor counts, bool from_recv);
};
} // namespace teco_ep
