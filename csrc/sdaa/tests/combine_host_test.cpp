#include "combine.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

void combine_layout_f16() asm("_ZN6tecoep7kernels33__device_stub__get_combine_layoutIDF16_EEvPcmiiiPKT_PKi");
void combine_layout_lowlatency_f16() asm("_ZN6tecoep7kernels44__device_stub__get_combine_layout_lowlatencyIDF16_EEvPcmiiiPKT_PKiPm");
void combine_gather_f16() asm("_ZN6tecoep7kernels34__device_stub__gather_combine_dataIDF16_EEvPcS2_miiiiiiiPKiS4_PT_");
void combine_gather_lowlatency_f16() asm("_ZN6tecoep7kernels45__device_stub__gather_combine_data_lowlatencyIDF16_EEvPcS2_miiiiiiiPKiS4_PT_PmS2_");
void combine_allreduce_f16() asm("_ZN6tecoep7kernels37__device_stub__combine_card_allreduceIDF16_Lm4096EEEvPPvS2_S2_miiiPKi");
void combine_layout_f16() {}
void combine_layout_lowlatency_f16() {}
void combine_gather_f16() {}
void combine_gather_lowlatency_f16() {}
void combine_allreduce_f16() {}

namespace {
std::vector<unsigned char> bytes;
const void* entry;
unsigned int push_count, pop_count, launch_count;
sdaaError_t push_status = sdaaSuccess;
auto stream = reinterpret_cast<sdaaStream_t>(0x1230);
void require(bool condition) { if (!condition) std::abort(); }
template<class T> T field(std::size_t offset) { T value; std::memcpy(&value, bytes.data() + offset, sizeof(value)); return value; }
void* ptr(std::uintptr_t value) { return reinterpret_cast<void*>(value); }
}
extern "C" {
sdaaError_t __sdaaPushCallConfiguration(std::size_t shared, sdaaStream_t actual_stream) {
    ++push_count; require(shared == 1 && actual_stream == stream); return push_status;
}
sdaaError_t __sdaaPopCallConfiguration(std::size_t* shared, sdaaStream_t* actual_stream) {
    ++pop_count; *shared = 17; *actual_stream = reinterpret_cast<sdaaStream_t>(0x4560); return sdaaErrorInvalidValue;
}
sdaaError_t sdaaLaunchKernel(const void* actual_entry, void** args, std::size_t shared, sdaaStream_t actual_stream) {
    ++launch_count; entry = actual_entry;
    require(args[0] == SDAA_LAUNCH_PARAM_BUFFER_POINTER && args[2] == SDAA_LAUNCH_PARAM_BUFFER_SIZE && args[4] == SDAA_LAUNCH_PARAM_END);
    require(shared == 17 && actual_stream == reinterpret_cast<sdaaStream_t>(0x4560));
    const auto size = reinterpret_cast<std::uintptr_t>(args[3]);
    auto* first = static_cast<unsigned char*>(args[1]); bytes.assign(first, first + size);
    return sdaaErrorInvalidValue;
}
}
int main() {
    using namespace tecoep::kernels;
    auto layout = [&](int dp, int ep, EP_DTYPE dtype) {
        return get_combine_layout(stream, ptr(0x1000), 0x100000003ul, dp, ep, 5120, ptr(0x2000), ptr(0x3000), dtype);
    };
    for (unsigned int dtype = 0; dtype < 6; ++dtype) {
        const auto before = push_count;
        require(layout(32, 32, static_cast<EP_DTYPE>(dtype)) == (dtype == 0 ? EP_RETURN_SUCCESS : EP_RETURN_FAILED));
        require(push_count == before + (dtype == 0));
    }
    require(layout(33, 32, EP_DTYPE_FLOAT16) == EP_RETURN_FAILED);
    require(layout(32, 33, EP_DTYPE_FLOAT16) == EP_RETURN_FAILED);
    require(layout(-1, -2, EP_DTYPE_FLOAT16) == EP_RETURN_SUCCESS);
    require(bytes.size() == 48 && entry == reinterpret_cast<void*>(combine_layout_f16));
    require(field<std::uintptr_t>(0) == 0x1000 && field<unsigned long>(8) == 0x100000003ul);
    require(field<int>(16) == -1 && field<int>(20) == -2 && field<int>(24) == 5120);
    require(field<std::uintptr_t>(32) == 0x2000 && field<std::uintptr_t>(40) == 0x3000);
    auto* barrier = reinterpret_cast<unsigned long*>(0x6000);
    require(get_combine_layout_lowlatency(stream, ptr(1), 2, 3, 4, 5, ptr(6), ptr(7), EP_DTYPE_FLOAT16, barrier) == EP_RETURN_SUCCESS);
    require(bytes.size() == 56 && field<std::uintptr_t>(48) == 0x6000);
    require(entry == reinterpret_cast<void*>(combine_layout_lowlatency_f16));
    auto gather = [&](int hidden, EP_DTYPE dtype) {
        return gather_combine_data(stream, ptr(0x10), ptr(0x20), 0x100000009ul, 2, 40, 80, 600, hidden, 3, 8, ptr(0x30), ptr(0x40), ptr(0x50), dtype);
    };
    for (int hidden : {0, 256, 5120, -256}) require(gather(hidden, EP_DTYPE_FLOAT16) == EP_RETURN_SUCCESS);
    for (int hidden : {1, 255, 257, 5119, -1}) {
        const auto before = push_count; require(gather(hidden, EP_DTYPE_FLOAT16) == EP_RETURN_FAILED); require(push_count == before);
    }
    require(gather(5120, EP_DTYPE_FLOAT32) == EP_RETURN_FAILED);
    require(gather(5120, EP_DTYPE_FLOAT16) == EP_RETURN_SUCCESS);
    require(bytes.size() == 80 && entry == reinterpret_cast<void*>(combine_gather_f16));
    require(field<unsigned long>(16) == 0x100000009ul && field<int>(24) == 2 && field<int>(28) == 40);
    require(field<int>(32) == 80 && field<int>(36) == 600 && field<int>(40) == 5120);
    require(field<int>(44) == 3 && field<int>(48) == 8);
    require(field<std::uintptr_t>(56) == 0x30 && field<std::uintptr_t>(64) == 0x40 && field<std::uintptr_t>(72) == 0x50);
    require(gather_combine_data_lowlatency(stream, ptr(1), ptr(2), 3, 4, 5, 6, 7, 5120, 9, 10, ptr(11), ptr(12), ptr(13), EP_DTYPE_FLOAT16, barrier, ptr(15)) == EP_RETURN_SUCCESS);
    require(bytes.size() == 96 && field<std::uintptr_t>(80) == 0x6000 && field<std::uintptr_t>(88) == 15);
    require(entry == reinterpret_cast<void*>(combine_gather_lowlatency_f16));
    require(combine_card_allreduce(stream, reinterpret_cast<void**>(0x70), ptr(0x80), ptr(0x90), 0x10000000bul, 2, 600, 5120, ptr(0xa0), EP_DTYPE_FLOAT16) == EP_RETURN_SUCCESS);
    require(bytes.size() == 56 && entry == reinterpret_cast<void*>(combine_allreduce_f16));
    require(field<std::uintptr_t>(0) == 0x70 && field<std::uintptr_t>(16) == 0x90 && field<unsigned long>(24) == 0x10000000bul);
    require(field<int>(32) == 2 && field<int>(36) == 600 && field<int>(40) == 5120 && field<std::uintptr_t>(48) == 0xa0);
    const auto before_pop = pop_count, before_launch = launch_count;
    push_status = sdaaErrorInvalidValue;
    require(layout(16, 16, EP_DTYPE_FLOAT16) == EP_RETURN_FAILED);
    require(pop_count == before_pop && launch_count == before_launch);
}
