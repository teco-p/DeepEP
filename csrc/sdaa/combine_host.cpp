#include "combine.hpp"

// Original generated entries, awaiting recovered device bodies and SCPP registration.
extern void combine_layout_f16() asm("_ZN6tecoep7kernels33__device_stub__get_combine_layoutIDF16_EEvPcmiiiPKT_PKi");
extern void combine_layout_lowlatency_f16() asm("_ZN6tecoep7kernels44__device_stub__get_combine_layout_lowlatencyIDF16_EEvPcmiiiPKT_PKiPm");
extern void combine_gather_f16() asm("_ZN6tecoep7kernels34__device_stub__gather_combine_dataIDF16_EEvPcS2_miiiiiiiPKiS4_PT_");
extern void combine_gather_lowlatency_f16() asm("_ZN6tecoep7kernels45__device_stub__gather_combine_data_lowlatencyIDF16_EEvPcS2_miiiiiiiPKiS4_PT_PmS2_");
extern void combine_allreduce_f16() asm("_ZN6tecoep7kernels37__device_stub__combine_card_allreduceIDF16_Lm4096EEEvPPvS2_S2_miiiPKi");

namespace {
struct LayoutArguments {
    void* send;
    std::size_t capacity;
    int dp, ep, hidden;
    const void* expert_hidden;
    const void* counts;
};
struct LowLatencyLayoutArguments { LayoutArguments layout; unsigned long* barrier; };
struct GatherArguments {
    void* send;
    void* recv;
    std::size_t capacity;
    int dp_rank, dp, ep, num_tokens, hidden, card_id, card_num;
    const void* mask;
    const void* counts;
    void* out;
};
struct LowLatencyGatherArguments { GatherArguments gather; unsigned long* barrier; void* workspace; };
struct AllreduceArguments {
    void** hidden_buffers;
    void* out;
    void* workspace;
    std::size_t capacity;
    int local_rank, hidden, dp;
    const void* counts;
};
static_assert(sizeof(LayoutArguments) == 48 && offsetof(LayoutArguments, expert_hidden) == 32);
static_assert(sizeof(LowLatencyLayoutArguments) == 56);
static_assert(sizeof(GatherArguments) == 80 && offsetof(GatherArguments, mask) == 56);
static_assert(sizeof(LowLatencyGatherArguments) == 96);
static_assert(sizeof(AllreduceArguments) == 56 && offsetof(AllreduceArguments, counts) == 48);

template<class Arguments>
EP_DISPATCH launch_combine(sdaaStream_t stream, void (*entry)(), Arguments& arguments) {
    if (__sdaaPushCallConfiguration(1, stream) != sdaaSuccess) return EP_RETURN_FAILED;
    void* launch_args[] = {SDAA_LAUNCH_PARAM_BUFFER_POINTER, &arguments,
        SDAA_LAUNCH_PARAM_BUFFER_SIZE, reinterpret_cast<void*>(sizeof(Arguments)),
        SDAA_LAUNCH_PARAM_END};
    std::size_t shared_mem;
    sdaaStream_t configured_stream;
    if (__sdaaPopCallConfiguration(&shared_mem, &configured_stream) != sdaaSuccess)
        return EP_RETURN_FAILED;
    return sdaaLaunchKernel(reinterpret_cast<const void*>(entry), launch_args,
        shared_mem, configured_stream) == sdaaSuccess ? EP_RETURN_SUCCESS : EP_RETURN_FAILED;
}
} // namespace

namespace tecoep::kernels {
// Aux ELF 0x2b10.
EP_DISPATCH get_combine_layout(sdaaStream_t stream, void* send_buffer,
    std::size_t capacity, int dp_size, int ep_size, int hidden_size,
    const void* expert_hidden, const void* counts, EP_DTYPE dtype) {
    if (dp_size > 32 || ep_size > 32 || dtype != EP_DTYPE_FLOAT16) return EP_RETURN_FAILED;
    LayoutArguments args{send_buffer, capacity, dp_size, ep_size, hidden_size, expert_hidden, counts};
    return launch_combine(stream, combine_layout_f16, args);
}
// Aux ELF 0x2c2c.
EP_DISPATCH get_combine_layout_lowlatency(sdaaStream_t stream, void* send_buffer,
    std::size_t capacity, int dp_size, int ep_size, int hidden_size,
    const void* expert_hidden, const void* counts, EP_DTYPE dtype, unsigned long* barrier) {
    if (dp_size > 32 || ep_size > 32 || dtype != EP_DTYPE_FLOAT16) return EP_RETURN_FAILED;
    LowLatencyLayoutArguments args{{send_buffer, capacity, dp_size, ep_size, hidden_size, expert_hidden, counts}, barrier};
    return launch_combine(stream, combine_layout_lowlatency_f16, args);
}
// Aux ELF 0x2338. Its hidden-size low-byte test is present before any runtime call.
EP_DISPATCH gather_combine_data(sdaaStream_t stream, void* send, void* recv,
    std::size_t capacity, int dp_rank, int dp_size, int ep_size, int num_tokens,
    int hidden_size, int card_id, int card_num, const void* mask,
    const void* counts, void* out, EP_DTYPE dtype) {
    if ((static_cast<unsigned int>(hidden_size) & 255u) != 0 || dtype != EP_DTYPE_FLOAT16) return EP_RETURN_FAILED;
    GatherArguments args{send, recv, capacity, dp_rank, dp_size, ep_size, num_tokens,
        hidden_size, card_id, card_num, mask, counts, out};
    return launch_combine(stream, combine_gather_f16, args);
}
// Aux ELF 0x24a0.
EP_DISPATCH gather_combine_data_lowlatency(sdaaStream_t stream, void* send, void* recv,
    std::size_t capacity, int dp_rank, int dp_size, int ep_size, int num_tokens,
    int hidden_size, int card_id, int card_num, const void* mask,
    const void* counts, void* out, EP_DTYPE dtype, unsigned long* barrier, void* workspace) {
    if ((static_cast<unsigned int>(hidden_size) & 255u) != 0 || dtype != EP_DTYPE_FLOAT16) return EP_RETURN_FAILED;
    LowLatencyGatherArguments args{{send, recv, capacity, dp_rank, dp_size, ep_size, num_tokens,
        hidden_size, card_id, card_num, mask, counts, out}, barrier, workspace};
    return launch_combine(stream, combine_gather_lowlatency_f16, args);
}
// Aux ELF 0x29f0; the sole device template has fixed 4096 block extent.
EP_DISPATCH combine_card_allreduce(sdaaStream_t stream, void** hidden_buffers,
    void* out, void* workspace, std::size_t capacity, int local_rank,
    int hidden_size, int dp_size, const void* counts, EP_DTYPE dtype) {
    if (dtype != EP_DTYPE_FLOAT16) return EP_RETURN_FAILED;
    AllreduceArguments args{hidden_buffers, out, workspace, capacity, local_rank, hidden_size, dp_size, counts};
    return launch_combine(stream, combine_allreduce_f16, args);
}
} // namespace tecoep::kernels
