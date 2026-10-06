#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <sdaa_runtime.h>

// Reconstructed original SDAA failure path: stdout diagnostic and exit(1).
inline void ep_sdaa_check(sdaaError_t status, const char* call,
                          const char* function, int original_line) {
    if (status == sdaaSuccess) return;
    std::printf("\033[31merror: '%s'(%d) from %s at %s:%d\033[0m\n",
        sdaaGetErrorString(status), static_cast<int>(status), call,
        function, original_line);
    std::exit(1);
}
