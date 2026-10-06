#pragma once

#include "dispatch.hpp"

// Signatures from the main extension's DWARF; device bodies remain unrecovered.
namespace tecoep::kernels {
EP_DISPATCH get_combine_layout(sdaaStream_t stream, void* send_buffer,
    std::size_t cross_buffer_size, int dp_size, int ep_size, int hidden_size,
    const void* expert_hidden, const void* recv_num_tokens_per_dp, EP_DTYPE dtype);
EP_DISPATCH gather_combine_data(sdaaStream_t stream, void* send_buffer,
    void* recv_buffer, std::size_t cross_buffer_size, int dp_rank, int dp_size,
    int ep_size, int num_tokens, int hidden_size, int card_id_per_dp,
    int card_num_per_dp, const void* is_token_in_card,
    const void* num_tokens_per_card, void* out_hidden_states, EP_DTYPE dtype);
EP_DISPATCH get_combine_layout_lowlatency(sdaaStream_t stream, void* send_buffer,
    std::size_t cross_buffer_size, int dp_size, int ep_size, int hidden_size,
    const void* expert_hidden, const void* recv_num_tokens_per_dp, EP_DTYPE dtype,
    unsigned long* barrier_flag);
EP_DISPATCH gather_combine_data_lowlatency(sdaaStream_t stream, void* send_buffer,
    void* recv_buffer, std::size_t cross_buffer_size, int dp_rank, int dp_size,
    int ep_size, int num_tokens, int hidden_size, int card_id_per_dp,
    int card_num_per_dp, const void* is_token_in_card,
    const void* num_tokens_per_card, void* out_hidden_states, EP_DTYPE dtype,
    unsigned long* barrier_flag, void* workspace);
EP_DISPATCH combine_card_allreduce(sdaaStream_t stream, void** hidden_buffer_ptrs,
    void* out_hidden_states, void* workspace, std::size_t cross_buffer_size,
    int rank_id_local_card, int hidden_size, int dp_size,
    const void* recv_num_tokens_per_dp, EP_DTYPE dtype);
} // namespace tecoep::kernels
