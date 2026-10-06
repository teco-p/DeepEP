#include "epoch.hpp"

namespace tecoep::kernels {
extern void __device_stub__wait_receive_epoch_kernel(unsigned long*,
    unsigned long*, int, int, int, std::int64_t*);

EP_DISPATCH wait_receive_epoch(sdaaStream_t stream, unsigned long* published,
    unsigned long* expected, int dp_rank, int dp_size, int phase,
    std::int64_t* status) {
    if (!published || !expected || !status || dp_size < 1 || dp_size > 32 ||
        dp_rank < 0 || dp_rank >= dp_size) return EP_RETURN_FAILED;
    struct Arguments {
        unsigned long* published;
        unsigned long* expected;
        int dp_rank, dp_size, phase;
        std::int64_t* status;
    } args{published, expected, dp_rank, dp_size, phase, status};
    static_assert(sizeof(Arguments) == 40 && offsetof(Arguments, status) == 32);
    if (__sdaaPushCallConfiguration(1, stream) != sdaaSuccess) return EP_RETURN_FAILED;
    void* launch_args[] = {SDAA_LAUNCH_PARAM_BUFFER_POINTER, &args,
        SDAA_LAUNCH_PARAM_BUFFER_SIZE, reinterpret_cast<void*>(sizeof(args)),
        SDAA_LAUNCH_PARAM_END};
    std::size_t shared_mem;
    sdaaStream_t configured_stream;
    if (__sdaaPopCallConfiguration(&shared_mem, &configured_stream) != sdaaSuccess)
        return EP_RETURN_FAILED;
    return sdaaLaunchKernel(reinterpret_cast<const void*>(
        &__device_stub__wait_receive_epoch_kernel), launch_args, shared_mem,
        configured_stream) == sdaaSuccess ? EP_RETURN_SUCCESS : EP_RETURN_FAILED;
}
}
