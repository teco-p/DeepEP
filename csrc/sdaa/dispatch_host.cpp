#include "dispatch.hpp"

// g447683d host functions recovered with LLVM 22.1.8 and Ghidra 12.1.4.
// Device implementations and their SCPP registration are still required.
// These are references to generated device entry points, not implementations.
extern void layout_f16() asm("_ZN6tecoep7kernels34__device_stub__get_dispatch_layoutIDF16_DF16_EEvPcmiiiiiiiiiiPiPT0_PT_S3_S3_");
extern void layout_f32() asm("_ZN6tecoep7kernels34__device_stub__get_dispatch_layoutIDF16_fEEvPcmiiiiiiiiiiPiPT0_PT_S3_S3_");
extern void layout_lowlatency_f16() asm("_ZN6tecoep7kernels45__device_stub__get_dispatch_layout_lowlatencyIDF16_DF16_EEvPcmiiiiiiiiiiPiPT0_PT_S3_S3_Pm");
extern void layout_lowlatency_f32() asm("_ZN6tecoep7kernels45__device_stub__get_dispatch_layout_lowlatencyIDF16_fEEvPcmiiiiiiiiiiPiPT0_PT_S3_S3_Pm");
extern void gather_f16() asm("_ZN6tecoep7kernels35__device_stub__gather_dispatch_dataIDF16_DF16_EEvPcS2_miiiiPiS3_PT0_PT_");
extern void gather_f32() asm("_ZN6tecoep7kernels35__device_stub__gather_dispatch_dataIDF16_fEEvPcS2_miiiiPiS3_PT0_PT_");
extern void gather_lowlatency_f16() asm("_ZN6tecoep7kernels46__device_stub__gather_dispatch_data_lowlatencyIDF16_DF16_EEvPcS2_miiiiPiS3_PT0_PT_Pm");
extern void gather_lowlatency_f32() asm("_ZN6tecoep7kernels46__device_stub__gather_dispatch_data_lowlatencyIDF16_fEEvPcS2_miiiiPiS3_PT0_PT_Pm");

namespace {

struct LayoutArguments {
    void* send_buffer;
    std::size_t cross_buffer_size;
    int dp_size;
    int card_id_per_dp;
    int card_num_per_dp;
    int rank_id_local_card;
    int ep_size;
    int num_tokens;
    int hidden_size;
    int topk;
    int local_expert_num;
    int global_expert_num;
    const void* topk_idx;
    const void* topk_weight;
    const void* hidden_states;
    void* is_token_in_card;
    void* num_tokens_per_card;
};

struct GatherArguments {
    void* send_buffer;
    void* recv_buffer;
    std::size_t cross_buffer_size;
    int dp_rank;
    int dp_size;
    int topk;
    int hidden_size;
    void* recv_num_tokens_per_dp;
    void* recv_topk_idx;
    void* recv_topk_weight;
    void* recv_hidden_states;
};

struct LowLatencyLayoutArguments {
    LayoutArguments layout;
    unsigned long* barrier_flag;
};

struct LowLatencyGatherArguments {
    GatherArguments gather;
    unsigned long* barrier_flag;
};

static_assert(sizeof(void*) == 8 && sizeof(int) == 4 && sizeof(unsigned long) == 8);
static_assert(sizeof(LayoutArguments) == 96);
static_assert(offsetof(LayoutArguments, topk_idx) == 56);
static_assert(offsetof(LayoutArguments, num_tokens_per_card) == 88);
static_assert(sizeof(GatherArguments) == 72);
static_assert(offsetof(GatherArguments, recv_num_tokens_per_dp) == 40);
static_assert(offsetof(GatherArguments, recv_hidden_states) == 64);
static_assert(sizeof(LowLatencyLayoutArguments) == 104);
static_assert(sizeof(LowLatencyGatherArguments) == 80);

bool supported_dtypes(EP_DTYPE hidden, EP_DTYPE weight) {
    return hidden == EP_DTYPE_FLOAT16 &&
        (weight == EP_DTYPE_FLOAT16 || weight == EP_DTYPE_FLOAT32);
}

template<class Arguments>
EP_DISPATCH launch_dispatch(sdaaStream_t stream, void (*entry)(), Arguments& arguments) {
    if (__sdaaPushCallConfiguration(1, stream) != sdaaSuccess) return EP_RETURN_FAILED;
    void* launch_args[] = {
        SDAA_LAUNCH_PARAM_BUFFER_POINTER,
        &arguments,
        SDAA_LAUNCH_PARAM_BUFFER_SIZE,
        reinterpret_cast<void*>(sizeof(Arguments)),
        SDAA_LAUNCH_PARAM_END,
    };
    std::size_t shared_mem;
    sdaaStream_t configured_stream;
    if (__sdaaPopCallConfiguration(&shared_mem, &configured_stream) != sdaaSuccess)
        return EP_RETURN_FAILED;
    return sdaaLaunchKernel(reinterpret_cast<const void*>(entry), launch_args,
        shared_mem, configured_stream) == sdaaSuccess ? EP_RETURN_SUCCESS : EP_RETURN_FAILED;
}

} // namespace

namespace tecoep::kernels {

// Original host address 0x2d50, size 660.
EP_DISPATCH get_dispatch_layout(sdaaStream_t stream, void* send_buffer,
    std::size_t cross_buffer_size, int dp_size, int card_id_per_dp,
    int card_num_per_dp, int rank_id_local_card, int ep_size,
    int num_tokens, int hidden_size, int topk, int local_expert_num,
    int global_expert_num, const void* topk_idx, const void* topk_weight,
    const void* hidden_states, void* is_token_in_card,
    void* num_tokens_per_card, EP_DTYPE hidden_dtype, EP_DTYPE weight_dtype) {
    // Signed comparisons in the original do not reject negative values.
    if (dp_size > 32 || ep_size > 32 || !supported_dtypes(hidden_dtype, weight_dtype)) return EP_RETURN_FAILED;
    LayoutArguments args{send_buffer, cross_buffer_size, dp_size, card_id_per_dp,
        card_num_per_dp, rank_id_local_card, ep_size, num_tokens, hidden_size,
        topk, local_expert_num, global_expert_num, topk_idx, topk_weight,
        hidden_states, is_token_in_card, num_tokens_per_card};
    return launch_dispatch(stream, weight_dtype == EP_DTYPE_FLOAT16 ? layout_f16 : layout_f32, args);
}

// Original host address 0x2fe4, size 684.
EP_DISPATCH get_dispatch_layout_lowlatency(sdaaStream_t stream, void* send_buffer,
    std::size_t cross_buffer_size, int dp_size, int card_id_per_dp,
    int card_num_per_dp, int rank_id_local_card, int ep_size,
    int num_tokens, int hidden_size, int topk, int local_expert_num,
    int global_expert_num, const void* topk_idx, const void* topk_weight,
    const void* hidden_states, void* is_token_in_card,
    void* num_tokens_per_card, EP_DTYPE hidden_dtype, EP_DTYPE weight_dtype,
    unsigned long* barrier_flag) {
    if (dp_size > 32 || ep_size > 32 || !supported_dtypes(hidden_dtype, weight_dtype)) return EP_RETURN_FAILED;
    LowLatencyLayoutArguments args{{send_buffer, cross_buffer_size, dp_size,
        card_id_per_dp, card_num_per_dp, rank_id_local_card, ep_size, num_tokens,
        hidden_size, topk, local_expert_num, global_expert_num, topk_idx,
        topk_weight, hidden_states, is_token_in_card, num_tokens_per_card}, barrier_flag};
    return launch_dispatch(stream,
        weight_dtype == EP_DTYPE_FLOAT16 ? layout_lowlatency_f16 : layout_lowlatency_f32, args);
}

// Original host address 0x2618, size 480.
EP_DISPATCH gather_dispatch_data(sdaaStream_t stream, void* send_buffer,
    void* recv_buffer, std::size_t cross_buffer_size, int dp_rank,
    int dp_size, int topk, int hidden_size, void* recv_num_tokens_per_dp,
    void* recv_topk_idx, void* recv_topk_weight, void* recv_hidden_states,
    EP_DTYPE hidden_dtype, EP_DTYPE weight_dtype) {
    if (!supported_dtypes(hidden_dtype, weight_dtype)) return EP_RETURN_FAILED;
    GatherArguments args{send_buffer, recv_buffer, cross_buffer_size, dp_rank,
        dp_size, topk, hidden_size, recv_num_tokens_per_dp, recv_topk_idx,
        recv_topk_weight, recv_hidden_states};
    return launch_dispatch(stream, weight_dtype == EP_DTYPE_FLOAT16 ? gather_f16 : gather_f32, args);
}

// Original host address 0x27f8, size 504.
EP_DISPATCH gather_dispatch_data_lowlatency(sdaaStream_t stream, void* send_buffer,
    void* recv_buffer, std::size_t cross_buffer_size, int dp_rank,
    int dp_size, int topk, int hidden_size, void* recv_num_tokens_per_dp,
    void* recv_topk_idx, void* recv_topk_weight, void* recv_hidden_states,
    EP_DTYPE hidden_dtype, EP_DTYPE weight_dtype, unsigned long* barrier_flag) {
    if (!supported_dtypes(hidden_dtype, weight_dtype)) return EP_RETURN_FAILED;
    LowLatencyGatherArguments args{{send_buffer, recv_buffer, cross_buffer_size,
        dp_rank, dp_size, topk, hidden_size, recv_num_tokens_per_dp,
        recv_topk_idx, recv_topk_weight, recv_hidden_states}, barrier_flag};
    return launch_dispatch(stream,
        weight_dtype == EP_DTYPE_FLOAT16 ? gather_lowlatency_f16 : gather_lowlatency_f32, args);
}

} // namespace tecoep::kernels
