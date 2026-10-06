#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_sdaa/sdaa_extension.h>
#include "combine.hpp"
#include "dispatch.hpp"
#include "exception.hpp"

namespace tecoep::ops {

// Main ELF 0x41be8. Sizes narrow to int before the kernel call, as in the original.
void get_dispatch_layout(const at::Tensor& topk_idx, const at::Tensor& topk_weight,
                         const at::Tensor& hidden_states, long local_expert_num,
                         long global_expert_num, long ep_size, long dp_size,
                         long card_id_per_dp, long card_num_per_dp, long rank_id_local_card,
                         long cross_buffer_size, at::Tensor& send_buffer_tensor,
                         at::Tensor& is_token_in_card, at::Tensor& num_tokens_per_card) {
    const int num_tokens = hidden_states.size(0);
    const int hidden_size = hidden_states.size(1);
    const int topk = topk_idx.size(1);
    const auto stream = torch::sdaa::getCurrentSDAAStream(-1);
    const auto status = kernels::get_dispatch_layout(
        stream, send_buffer_tensor.data_ptr(), cross_buffer_size, dp_size,
        card_id_per_dp, card_num_per_dp, rank_id_local_card, ep_size, num_tokens,
        hidden_size, topk, local_expert_num, global_expert_num, topk_idx.const_data_ptr(),
        topk_weight.const_data_ptr(), hidden_states.const_data_ptr(), is_token_in_card.data_ptr(),
        num_tokens_per_card.data_ptr(), EP_DTYPE_FLOAT16,
        topk_weight.scalar_type() == at::kHalf ? EP_DTYPE_FLOAT16 : EP_DTYPE_FLOAT32);
    if (status != EP_RETURN_SUCCESS) {
        throw EPException(
            "Failed to call: tecoep::kernels::get_dispatch_layout( stream, send_buffer_tensor.data_ptr(), cross_buffer_size, dp_size, card_id_per_dp, card_num_per_dp, rank_id_local_card, ep_size, num_tokens, hidden_size, topk, local_expert_num, global_expert_num, topk_idx.const_data_ptr(), topk_weight.const_data_ptr(), hidden_states.const_data_ptr(), is_token_in_card.data_ptr(), num_tokens_per_card.data_ptr(), EP_DTYPE_FLOAT16, topk_weight.dtype() == torch::kFloat16 ? EP_DTYPE_FLOAT16 : EP_DTYPE_FLOAT32)",
            "/data/ci_env/slave_loongson_tecoep_py312/workspace/loongson_build_tecoep_py312/usertestdir_3/tecoep/csrc/kernels/get_dispatch_layout_op.cpp", 25);
    }
}

// Main ELF 0x3efa8.
void gather_dispatch_data(const at::Tensor& send_buffer, const at::Tensor& recv_buffer,
                          long cross_buffer_size, long dp_rank, long dp_size,
                          at::Tensor& recv_num_tokens_per_dp, at::Tensor& recv_topk_idx,
                          at::Tensor& recv_topk_weight, at::Tensor& recv_hidden_states) {
    const auto stream = torch::sdaa::getCurrentSDAAStream(-1);
    const int topk = recv_topk_idx.size(1);
    const int hidden_size = recv_hidden_states.size(1);
    const auto status = kernels::gather_dispatch_data(
        stream, send_buffer.data_ptr(), recv_buffer.data_ptr(), cross_buffer_size, dp_rank,
        dp_size, topk, hidden_size, recv_num_tokens_per_dp.data_ptr(), recv_topk_idx.data_ptr(),
        recv_topk_weight.data_ptr(), recv_hidden_states.data_ptr(), EP_DTYPE_FLOAT16,
        recv_topk_weight.scalar_type() == at::kHalf ? EP_DTYPE_FLOAT16 : EP_DTYPE_FLOAT32);
    if (status != EP_RETURN_SUCCESS) {
        throw EPException(
            "Failed to call: tecoep::kernels::gather_dispatch_data( stream, send_buffer.data_ptr(), recv_buffer.data_ptr(), cross_buffer_size, dp_rank, dp_size, topk, hidden_size, recv_num_tokens_per_dp.data_ptr(), recv_topk_idx.data_ptr(), recv_topk_weight.data_ptr(), recv_hidden_states.data_ptr(), EP_DTYPE_FLOAT16, recv_topk_weight.dtype() == torch::kFloat16 ? EP_DTYPE_FLOAT16 : EP_DTYPE_FLOAT32)",
            "/data/ci_env/slave_loongson_tecoep_py312/workspace/loongson_build_tecoep_py312/usertestdir_3/tecoep/csrc/kernels/gather_dispatch_data_op.cpp", 19);
    }
}

// Main ELF 0x40528.
void get_combine_layout(const at::Tensor& expert_hidden, const at::Tensor& recv_num_tokens_per_dp,
                        long ep_size, long dp_size, long cross_buffer_size,
                        at::Tensor& send_buffer_tensor) {
    const int hidden_size = expert_hidden.size(1);
    const auto stream = torch::sdaa::getCurrentSDAAStream(-1);
    const auto status = kernels::get_combine_layout(
        stream, send_buffer_tensor.data_ptr(), cross_buffer_size, dp_size, ep_size, hidden_size,
        expert_hidden.const_data_ptr(), recv_num_tokens_per_dp.const_data_ptr(), EP_DTYPE_FLOAT16);
    if (status != EP_RETURN_SUCCESS) {
        throw EPException(
            "Failed to call: tecoep::kernels::get_combine_layout( stream, send_buffer_tensor.data_ptr(), cross_buffer_size, dp_size, ep_size, hidden_size, expert_hidden.const_data_ptr(), recv_num_tokens_per_dp.const_data_ptr(), EP_DTYPE_FLOAT16)",
            "/data/ci_env/slave_loongson_tecoep_py312/workspace/loongson_build_tecoep_py312/usertestdir_3/tecoep/csrc/kernels/get_combine_layout_op.cpp", 16);
    }
}

// Main ELF 0x3d2e8.
void gather_combine_data(const at::Tensor& send_buffer, const at::Tensor& recv_buffer,
                        long cross_buffer_size, long dp_rank, long dp_size, long ep_size,
                        const at::Tensor& is_token_in_card, const at::Tensor& num_tokens_per_card,
                        long card_id_per_dp, long card_num_per_dp, at::Tensor& out_hidden_states) {
    const auto stream = torch::sdaa::getCurrentSDAAStream(-1);
    const int num_tokens = out_hidden_states.size(0);
    const int hidden_size = out_hidden_states.size(1);
    const auto status = kernels::gather_combine_data(
        stream, send_buffer.data_ptr(), recv_buffer.data_ptr(), static_cast<std::size_t>(cross_buffer_size),
        static_cast<int>(dp_rank), static_cast<int>(dp_size), static_cast<int>(ep_size), num_tokens,
        hidden_size, static_cast<int>(card_id_per_dp), static_cast<int>(card_num_per_dp),
        is_token_in_card.const_data_ptr(), num_tokens_per_card.const_data_ptr(),
        out_hidden_states.data_ptr(), EP_DTYPE_FLOAT16);
    if (status != EP_RETURN_SUCCESS) {
        throw EPException(
            "Failed to call: tecoep::kernels::gather_combine_data( stream, send_buffer.data_ptr(), recv_buffer.data_ptr(), static_cast<size_t>(cross_buffer_size), static_cast<int>(dp_rank), static_cast<int>(dp_size), static_cast<int>(ep_size), num_tokens, hidden_size, static_cast<int>(card_id_per_dp), static_cast<int>(card_num_per_dp), is_token_in_card.const_data_ptr(), num_tokens_per_card.const_data_ptr(), out_hidden_states.data_ptr(), EP_DTYPE_FLOAT16)",
            "/data/ci_env/slave_loongson_tecoep_py312/workspace/loongson_build_tecoep_py312/usertestdir_3/tecoep/csrc/kernels/gather_combine_data_op.cpp", 21);
    }
}
} // namespace tecoep::ops

// Main ELF 0x461b8. 0x3e in its DWARF is c10::DispatchKey::PrivateUse1.
TORCH_LIBRARY(_ep_C, m) {
    m.def("get_dispatch_layout(Tensor topk_idx, Tensor topk_weight, Tensor hidden_states, int local_expert_num, int global_expert_num, int ep_size, int dp_size,  int card_id_per_dp, int card_num_per_dp, int rank_id_local_card, int cross_buffer_size, Tensor! send_buffer_tensor, Tensor! is_token_in_card, Tensor! num_tokens_per_card) -> ()");
    m.impl("get_dispatch_layout", torch::dispatch(c10::DispatchKey::PrivateUse1, TORCH_FN(tecoep::ops::get_dispatch_layout)));
    m.def("gather_dispatch_data(Tensor send_buffer, Tensor recv_buffer, int cross_buffer_size, int dp_rank, int dp_size, Tensor! recv_num_tokens_per_dp, Tensor! recv_topk_idx, Tensor! recv_topk_weight, Tensor! recv_hidden_states) -> ()");
    m.impl("gather_dispatch_data", torch::dispatch(c10::DispatchKey::PrivateUse1, TORCH_FN(tecoep::ops::gather_dispatch_data)));
    m.def("get_combine_layout(Tensor expert_hidden, Tensor recv_num_tokens_per_dp, int ep_size, int dp_size, int cross_buffer_size, Tensor! send_buffer) -> ()");
    m.impl("get_combine_layout", torch::dispatch(c10::DispatchKey::PrivateUse1, TORCH_FN(tecoep::ops::get_combine_layout)));
    m.def("gather_combine_data(Tensor send_buffer, Tensor recv_buffer, int cross_buffer_size, int dp_rank, int dp_size, int ep_size, Tensor is_token_in_card, Tensor num_tokens_per_card, int card_id_per_dp, int card_num_per_dp, Tensor! out_hidden_states) -> ()");
    m.impl("gather_combine_data", torch::dispatch(c10::DispatchKey::PrivateUse1, TORCH_FN(tecoep::ops::gather_combine_data)));
}
