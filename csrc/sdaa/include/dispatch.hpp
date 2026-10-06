#pragma once

#include <cstddef>
#include <cstdint>
#include <sdaa_runtime.h>
#include "ep_types.hpp"

namespace tecoep::kernels {

EP_DISPATCH get_dispatch_layout(sdaaStream_t stream, void* send_buffer,
    std::size_t cross_buffer_size, int dp_size, int card_id_per_dp,
    int card_num_per_dp, int rank_id_local_card, int ep_size,
    int num_tokens, int hidden_size, int topk, int local_expert_num,
    int global_expert_num, const void* topk_idx, const void* topk_weight,
    const void* hidden_states, void* is_token_in_card,
    void* num_tokens_per_card, EP_DTYPE hidden_dtype, EP_DTYPE weight_dtype);

EP_DISPATCH get_dispatch_layout_lowlatency(sdaaStream_t stream, void* send_buffer,
    std::size_t cross_buffer_size, int dp_size, int card_id_per_dp,
    int card_num_per_dp, int rank_id_local_card, int ep_size,
    int num_tokens, int hidden_size, int topk, int local_expert_num,
    int global_expert_num, const void* topk_idx, const void* topk_weight,
    const void* hidden_states, void* is_token_in_card,
    void* num_tokens_per_card, EP_DTYPE hidden_dtype, EP_DTYPE weight_dtype,
    unsigned long* barrier_flag);

EP_DISPATCH gather_dispatch_data(sdaaStream_t stream, void* send_buffer,
    void* recv_buffer, std::size_t cross_buffer_size, int dp_rank,
    int dp_size, int topk, int hidden_size, void* recv_num_tokens_per_dp,
    void* recv_topk_idx, void* recv_topk_weight, void* recv_hidden_states,
    EP_DTYPE hidden_dtype, EP_DTYPE weight_dtype);

EP_DISPATCH gather_dispatch_data_lowlatency(sdaaStream_t stream, void* send_buffer,
    void* recv_buffer, std::size_t cross_buffer_size, int dp_rank,
    int dp_size, int topk, int hidden_size, void* recv_num_tokens_per_dp,
    void* recv_topk_idx, void* recv_topk_weight, void* recv_hidden_states,
    EP_DTYPE hidden_dtype, EP_DTYPE weight_dtype, unsigned long* barrier_flag);

} // namespace tecoep::kernels
