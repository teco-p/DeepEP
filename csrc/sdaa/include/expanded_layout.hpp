#pragma once
#include <cstdint>

// The actual device producer and focused host receiver share this one layout.
// The caller supplies only the device function qualifier, never a second algorithm.
#ifndef DEEP_EP_LAYOUT_INLINE
#define DEEP_EP_LAYOUT_INLINE inline
#endif
namespace deep_ep::sdaa_backend {
DEEP_EP_LAYOUT_INLINE void layout_error(int64_t* status, int64_t code,
                                       int64_t item, int64_t value, int64_t bound) {
    if (status[0] == 0) {
        status[1] = item;
        status[2] = value;
        status[3] = bound;
        status[0] = code;
    }
}
// Port of fixed Ascend dispatch_copy_epilogue's expert-major slot allocation:
// each local top-k slot gets a destination, including repeated expert IDs.
DEEP_EP_LAYOUT_INLINE int build_expanded_layout(const uint16_t* hidden, const int* ids,
    const int* counts, int* rank_prefix, int* expert_prefix, int* expert_counts,
    int* padded_counts, int* route_slots, int* source, int capacity,
    int expanded_capacity, int width, int rank, int alignment, int64_t* status) {
    for (int row = 0; row < capacity; ++row) {
        for (int k = 0; k < 6; ++k) route_slots[row * 6 + k] = -1;
        source[row * 2] = source[row * 2 + 1] = -1;
    }
    for (int i = 0; i < 24; ++i) {
        expert_prefix[i] = expert_counts[i] = padded_counts[i] = 0;
    }
    for (int i = 0; i < 16; ++i) rank_prefix[i] = 0;
    if (status[0] != 0) return -1;
    int local_counts[24] = {};
    int cursors[24] = {};
    const int first = rank * 24;
    int total = 0, deduplicated = 0;
    for (int dp = 0; dp < 4; ++dp) {
        if (counts[dp] < 0 || counts[dp] > capacity - total) {
            layout_error(status, 5, dp, counts[dp], capacity - total);
            return -1;
        }
        const int end = total + counts[dp];
        for (int row = total; row < end; ++row) {
            bool present = false;
            for (int k = 0; k < 6; ++k) {
                const int id = ids[row * 6 + k];
                if (id < -1 || id >= 384) {
                    layout_error(status, 6, row * 6 + k, id, 384);
                    return -1;
                }
                if (id >= first && id < first + 24) {
                    ++local_counts[id - first];
                    present = true;
                }
            }
            if (present) {
                const auto source_row = hidden[
                    static_cast<long>(row) * (width + 16) + width];
                if (source_row >= capacity / 4) {
                    layout_error(status, 7, row, source_row, capacity / 4);
                    return -1;
                }
                source[row * 2] = dp * 4;
                source[row * 2 + 1] = source_row;
                ++deduplicated;
            }
        }
        for (int tp = 0; tp < 4; ++tp) rank_prefix[dp * 4 + tp] = deduplicated;
        total = end;
    }
    if (counts[4] != total) {
        layout_error(status, 5, 4, counts[4], total);
        return -1;
    }
    int offset = 0;
    for (int expert = 0; expert < 24; ++expert) {
        const int actual = local_counts[expert];
        const int padded = (actual + alignment - 1) & -alignment;
        if (padded > expanded_capacity - offset) {
            layout_error(status, 8, expert, padded, expanded_capacity - offset);
            return -1;
        }
        cursors[expert] = offset;
        expert_counts[expert] = actual;
        padded_counts[expert] = padded;
        // Ascend expanded prefix is aligned preceding start + actual count.
        expert_prefix[expert] = offset + actual;
        offset += padded;
    }
    for (int row = 0; row < total; ++row) {
        for (int k = 0; k < 6; ++k) {
            const int local = ids[row * 6 + k] - first;
            if (local >= 0 && local < 24)
                route_slots[row * 6 + k] = cursors[local]++;
        }
    }
    return offset;
}
}
