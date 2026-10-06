#pragma once
#include "dispatch.hpp"

namespace tecoep::kernels {
// Expected epochs are private to each SPA, persistent, and advanced by the
// device on every execution. status is sticky [phase, peer, expected, observed].
EP_DISPATCH wait_receive_epoch(sdaaStream_t stream, unsigned long* published,
    unsigned long* expected, int dp_rank, int dp_size, int phase,
    std::int64_t* status);
}
