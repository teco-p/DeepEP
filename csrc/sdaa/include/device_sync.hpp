#pragma once
#include <sdaa_runtime.h>
#include <sdaa_atomic.h>
#include <cstdint>

namespace tecoep::kernels {
// Original 0x33f0 helper and the identical inlined card-allreduce barrier.
// Fixed LLVM CAS lowering packs compare in word1 and new in word2.
__device__ inline void sync_card(long* flag, int participants) {
    if (RPEN() == 0) {
        int64_t* counter = reinterpret_cast<int64_t*>(flag);
        int64_t* epoch = counter + 1;
        volatile int64_t* epoch_read = epoch;
        int64_t initial;
        do {
            initial = *epoch_read;
        } while (!sdaa::atomic_cas_bool(epoch, initial, initial));
        (void)sdaa::atomic_inc(counter);
        if (sdaa::atomic_cas_bool(counter, static_cast<int64_t>(participants), int64_t{0})) {
            (void)sdaa::atomic_inc(epoch);
        }
        const unsigned long start = RTC();
        unsigned long now;
        int64_t current;
        do {
            now = RTC();
            do {
                current = *epoch_read;
            } while (!sdaa::atomic_cas_bool(epoch, current, current));
        } while (current == initial && now - start < 20000000000UL);
    }
    sdaa::sync_threads();
}
} // namespace tecoep::kernels
