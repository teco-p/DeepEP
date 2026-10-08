#include "buffer.hpp"
#include "combine.hpp"
#include "dispatch.hpp"
#include "epoch.hpp"
#include "exception.hpp"
#include "runtime_check.hpp"
#include <torch_sdaa/sdaa_extension.h>

namespace teco_ep {
namespace {
constexpr const char* original_file = "/data/ci_env/slave_loongson_tecoep_py312/workspace/loongson_build_tecoep_py312/usertestdir_3/tecoep/csrc/teco_ep.cpp";
void kernel_check(EP_DISPATCH status, const char* expression, int line) {
    if (status != EP_RETURN_SUCCESS) throw EPException(std::string("Failed to call: ") + expression, original_file, line);
}
}

// Main ELF 0x14bc0. The tensor's device query has its original observable errors.
void Buffer::get_dispatch_layout(const at::Tensor& topk_idx, const at::Tensor& topk_weight,
    const at::Tensor& hidden_states, int local_expert_num, int global_expert_num,
    int ep_size, int dp_size, at::Tensor& is_token_in_card, at::Tensor& num_tokens_per_card) {
    (void)hidden_states.device();
    const int num_tokens = hidden_states.size(0), hidden_size = hidden_states.size(1);
    const int topk = topk_idx.size(1);
    const std::size_t token_bytes = hidden_states.element_size() * hidden_size
        + (topk_idx.element_size() + topk_weight.element_size()) * topk;
    const std::size_t chunk = cross_buffer_size / static_cast<std::size_t>(this->dp_size);
    if (token_bytes == 0 || chunk < 4 || num_tokens < 0 ||
        static_cast<std::size_t>(num_tokens) > (chunk - 4) / token_bytes)
        throw EPException("dispatch shape bound exceeds transport chunk", original_file, 563);
    const auto stream = torch::sdaa::getCurrentSDAAStream(-1);
    void* send = dispatch.send_buffer_ptrs[device_info.card_id_local_node];
    const EP_DTYPE weight_dtype = topk_weight.scalar_type() == at::kHalf ? EP_DTYPE_FLOAT16 : EP_DTYPE_FLOAT32;
    if (!low_latency_mode) {
        kernel_check(tecoep::kernels::get_dispatch_layout(stream, send, cross_buffer_size,
            dp_size, device_info.card_id_per_dp, device_info.card_num_per_dp,
            device_info.rank_id_local_card, ep_size, num_tokens, hidden_size, topk,
            local_expert_num, global_expert_num, topk_idx.const_data_ptr(), topk_weight.const_data_ptr(),
            hidden_states.const_data_ptr(), is_token_in_card.data_ptr(), num_tokens_per_card.data_ptr(),
            EP_DTYPE_FLOAT16, weight_dtype),
            "tecoep::kernels::get_dispatch_layout( stream, dispatch.send_buffer_ptrs[device_info.card_id_local_node], cross_buffer_size, dp_size, device_info.card_id_per_dp, device_info.card_num_per_dp, device_info.rank_id_local_card, ep_size, num_tokens, hidden_size, topk, local_expert_num, global_expert_num, topk_idx.const_data_ptr(), topk_weight.const_data_ptr(), hidden_states.const_data_ptr(), is_token_in_card.data_ptr(), num_tokens_per_card.data_ptr(), EP_DTYPE_FLOAT16, topk_weight.dtype() == torch::kFloat16 ? EP_DTYPE_FLOAT16 : EP_DTYPE_FLOAT32)", 563);
    } else {
        kernel_check(tecoep::kernels::get_dispatch_layout_lowlatency(stream, send, cross_buffer_size,
            dp_size, device_info.card_id_per_dp, device_info.card_num_per_dp,
            device_info.rank_id_local_card, ep_size, num_tokens, hidden_size, topk,
            local_expert_num, global_expert_num, topk_idx.const_data_ptr(), topk_weight.const_data_ptr(),
            hidden_states.const_data_ptr(), is_token_in_card.data_ptr(), num_tokens_per_card.data_ptr(),
            EP_DTYPE_FLOAT16, weight_dtype, barrier_flag_),
            "tecoep::kernels::get_dispatch_layout_lowlatency( stream, dispatch.send_buffer_ptrs[device_info.card_id_local_node], cross_buffer_size, dp_size, device_info.card_id_per_dp, device_info.card_num_per_dp, device_info.rank_id_local_card, ep_size, num_tokens, hidden_size, topk, local_expert_num, global_expert_num, topk_idx.const_data_ptr(), topk_weight.const_data_ptr(), hidden_states.const_data_ptr(), is_token_in_card.data_ptr(), num_tokens_per_card.data_ptr(), EP_DTYPE_FLOAT16, topk_weight.dtype() == torch::kFloat16 ? EP_DTYPE_FLOAT16 : EP_DTYPE_FLOAT32, barrier_flag_)", 535);
    }
}

// Main ELF 0x151b0. This counts-size check precedes the stream query.
void Buffer::gather_dispatch_data(long dp_rank, long dp_size, const at::Tensor& counts,
    const at::Tensor& indices, const at::Tensor& weights, const at::Tensor& hidden) {
    if (dp_size + 1 != counts.size(0)) throw EPException(
        "dp_size + 1 == recv_num_tokens_per_dp.size(0)failed, recv_num_tokens_per_dp.size(0) must be equal to dp_size + 1.", original_file, 601);
    (void)hidden.device();
    const auto stream = torch::sdaa::getCurrentSDAAStream(-1);
    const int topk = indices.size(1), hidden_size = hidden.size(1);
    void* send = dispatch.send_buffer_ptrs[device_info.card_id_local_node];
    void* recv = dispatch.recv_buffer_ptrs[device_info.card_id_local_node];
    const EP_DTYPE weight_dtype = weights.scalar_type() == at::kHalf ? EP_DTYPE_FLOAT16 : EP_DTYPE_FLOAT32;
    if (!low_latency_mode) {
        kernel_check(tecoep::kernels::gather_dispatch_data(stream, send, recv, cross_buffer_size,
            dp_rank, dp_size, topk, hidden_size, counts.data_ptr(), indices.data_ptr(), weights.data_ptr(),
            hidden.data_ptr(), EP_DTYPE_FLOAT16, weight_dtype),
            "tecoep::kernels::gather_dispatch_data( stream, dispatch.send_buffer_ptrs[device_info.card_id_local_node], dispatch.recv_buffer_ptrs[device_info.card_id_local_node], cross_buffer_size, dp_rank, dp_size, topk, hidden_size, recv_num_tokens_per_dp.data_ptr(), recv_topk_idx.data_ptr(), recv_topk_weight.data_ptr(), recv_hidden_states.data_ptr(), EP_DTYPE_FLOAT16, recv_topk_weight.dtype() == torch::kFloat16 ? EP_DTYPE_FLOAT16 : EP_DTYPE_FLOAT32)", 629);
    } else {
        kernel_check(tecoep::kernels::gather_dispatch_data_lowlatency(stream, send, recv, cross_buffer_size,
            dp_rank, dp_size, topk, hidden_size, counts.data_ptr(), indices.data_ptr(), weights.data_ptr(),
            hidden.data_ptr(), EP_DTYPE_FLOAT16, weight_dtype, barrier_flag_ + dp_size),
            "tecoep::kernels::gather_dispatch_data_lowlatency( stream, dispatch.send_buffer_ptrs[device_info.card_id_local_node], dispatch.recv_buffer_ptrs[device_info.card_id_local_node], cross_buffer_size, dp_rank, dp_size, topk, hidden_size, recv_num_tokens_per_dp.data_ptr(), recv_topk_idx.data_ptr(), recv_topk_weight.data_ptr(), recv_hidden_states.data_ptr(), EP_DTYPE_FLOAT16, recv_topk_weight.dtype() == torch::kFloat16 ? EP_DTYPE_FLOAT16 : EP_DTYPE_FLOAT32, barrier_flag_ + dp_size)", 609);
    }
}

// Main ELF 0x15708. Ordinary mode publishes each SPA's expert pointer and uses
// the card allreduce; low latency mode runs a layout kernel on the leader only.
void Buffer::get_combine_layout(const at::Tensor& expert_hidden, const at::Tensor& counts,
    int ep_size, int dp_size) {
    const auto stream = torch::sdaa::getCurrentSDAAStream(-1);
    const int hidden_size = expert_hidden.size(1);
    void* send = combine.send_buffer_ptrs[device_info.card_id_local_node];
    if (!low_latency_mode) {
        ep_sdaa_check(sdaaStreamWriteValue64(stream, &combine.hidden_buffer_ptrs[device_info.rank_id_local_card],
            reinterpret_cast<unsigned long>(expert_hidden.data_ptr()), SDAA_STREAM_WRITE_VALUE_DEFAULT),
            "sdaaStreamWriteValue64( stream, &combine.hidden_buffer_ptrs[device_info.rank_id_local_card], (uint64_t)expert_hidden.data_ptr(), SDAA_STREAM_WRITE_VALUE_DEFAULT)", "get_combine_layout", 852);
        kernel_check(tecoep::kernels::combine_card_allreduce(stream, combine.hidden_buffer_ptrs,
            send, combine.spa_barrier_ptrs, cross_buffer_size, device_info.rank_id_local_card,
            hidden_size, dp_size, counts.const_data_ptr(), EP_DTYPE_FLOAT16),
            "tecoep::kernels::combine_card_allreduce( stream, combine.hidden_buffer_ptrs, combine.send_buffer_ptrs[device_info.card_id_local_node], combine.spa_barrier_ptrs, cross_buffer_size, device_info.rank_id_local_card, hidden_size, dp_size, recv_num_tokens_per_dp.const_data_ptr(), EP_DTYPE_FLOAT16)", 859);
    } else if (device_info.rank_id_local_card == 0) {
        kernel_check(tecoep::kernels::get_combine_layout_lowlatency(stream, send, cross_buffer_size,
            dp_size, ep_size, hidden_size, expert_hidden.const_data_ptr(), counts.const_data_ptr(),
            EP_DTYPE_FLOAT16, barrier_flag_ + 2 * dp_size),
            "tecoep::kernels::get_combine_layout_lowlatency( stream, combine.send_buffer_ptrs[device_info.card_id_local_node], cross_buffer_size, dp_size, ep_size, hidden_size, expert_hidden.const_data_ptr(), recv_num_tokens_per_dp.const_data_ptr(), EP_DTYPE_FLOAT16, barrier_flag_ + 2 * dp_size )", 874);
    }
}

// Main ELF 0x15a78. Ordinary mode waits in rotated DP order before the leader
// gathers; low latency device code owns its synchronization flags.
void Buffer::gather_combine_data(long dp_rank, long dp_size, long ep_size,
    const at::Tensor& mask, const at::Tensor& counts, at::Tensor& out) {
    const auto stream = torch::sdaa::getCurrentSDAAStream(-1);
    const int card = device_info.card_id_local_node;
    if (!low_latency_mode) {
        kernel_check(tecoep::kernels::wait_receive_epoch(stream, combine.recv_barrier_ptrs[card],
            reinterpret_cast<unsigned long*>(combine_expected_epoch_.data_ptr<std::int64_t>()),
            this->dp_rank, dp_size, 2, epoch_status_.data_ptr<std::int64_t>()),
            "wait_receive_epoch(combine)", 903);
    }
    if (device_info.rank_id_local_card != 0) return;
    const int num_tokens = out.size(0), hidden_size = out.size(1);
    if (!low_latency_mode) {
        kernel_check(tecoep::kernels::gather_combine_data(stream, combine.send_buffer_ptrs[card],
            combine.recv_buffer_ptrs[card], cross_buffer_size, dp_rank, dp_size, ep_size, num_tokens,
            hidden_size, device_info.card_id_per_dp, device_info.card_num_per_dp,
            mask.const_data_ptr(), counts.const_data_ptr(), out.data_ptr(), EP_DTYPE_FLOAT16),
            "tecoep::kernels::gather_combine_data( stream, combine.send_buffer_ptrs[device_info.card_id_local_node], combine.recv_buffer_ptrs[device_info.card_id_local_node], cross_buffer_size, (int)dp_rank, (int)dp_size, (int)ep_size, (int)out_hidden_states.size(0), (int)out_hidden_states.size(1), device_info.card_id_per_dp, device_info.card_num_per_dp, is_token_in_card.const_data_ptr(), num_tokens_per_card.const_data_ptr(), out_hidden_states.data_ptr(), EP_DTYPE_FLOAT16)", 932);
    } else {
        kernel_check(tecoep::kernels::gather_combine_data_lowlatency(stream, combine.send_buffer_ptrs[card],
            combine.recv_buffer_ptrs[card], cross_buffer_size, dp_rank, dp_size, ep_size, num_tokens,
            hidden_size, device_info.card_id_per_dp, device_info.card_num_per_dp,
            mask.const_data_ptr(), counts.const_data_ptr(), out.data_ptr(), EP_DTYPE_FLOAT16,
            barrier_flag_ + 3 * dp_size, workspace_),
            "tecoep::kernels::gather_combine_data_lowlatency( stream, combine.send_buffer_ptrs[device_info.card_id_local_node], combine.recv_buffer_ptrs[device_info.card_id_local_node], cross_buffer_size, (int)dp_rank, (int)dp_size, (int)ep_size, (int)out_hidden_states.size(0), (int)out_hidden_states.size(1), device_info.card_id_per_dp, device_info.card_num_per_dp, is_token_in_card.const_data_ptr(), num_tokens_per_card.const_data_ptr(), out_hidden_states.data_ptr(), EP_DTYPE_FLOAT16, barrier_flag_ + 3 * dp_size, workspace_)", 913);
    }
}
} // namespace teco_ep
