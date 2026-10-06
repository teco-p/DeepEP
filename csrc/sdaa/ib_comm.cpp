#include "ib_comm.hpp"
#include <cstdlib>

namespace {
// Inlined poll_prev in the original ib_comm.cpp; posting the next generation
// first consumes the completion count retained by the previous callback.
void poll_prev(ibv_tools::ibvResources* res) {
    if (res && res->pending_poll_size > 0) {
        ibv_tools::poll_completion(res, res->pending_poll_size, -1);
        res->pending_poll_size = 0;
    }
}
}

// Main ELF 0x43860. poll_size is not read by the original instructions.
void rdmaPostIbFunc(void* args) {
    auto* data = static_cast<tecoEPIbArgs*>(args);
    const std::size_t size = data->num_tokens_ptr
        ? static_cast<std::size_t>(*data->num_tokens_ptr) * data->data_length_per_token
        : data->data_size;
    ibv_tools::post_write(data->res, data->offset, data->dst_offset, size, data->peer);
    if (data->if_poll) ibv_tools::poll_completion(data->res, 1, data->peer);
    std::free(args);
}

// Main ELF 0x43930. data_size entries already contain packed byte counts.
void rdmaPostPollIBFuncData(void* args) {
    auto* data = static_cast<tecoEPIBArgsL*>(args);
    auto* res = data->res;
    poll_prev(res);
    int posted = 0;
    for (int i = 0; i < data->dp_size; ++i) {
        const int dst = data->inter_node_dst[i];
        if (dst != -1) {
            const std::size_t chunk = data->cross_buffer_size / static_cast<std::size_t>(data->dp_size);
            if (ibv_tools::post_write(res, i * chunk, data->dp_rank * chunk,
                    static_cast<std::size_t>(data->data_size[dst]), i) == 0) ++posted;
        }
    }
    res->pending_poll_size = posted;
    std::free(args);
}

// Main ELF 0x43a58. The four-byte count header precedes token data; the
// separate eight-byte barrier is posted after that peer's payload.
void rdmaPostPollIBFunc(void* args) {
    auto* data = static_cast<tecoEPIBArgsV*>(args);
    auto* res = data->res;
    const std::size_t chunk = data->cross_buffer_size / static_cast<std::size_t>(data->dp_size);
    poll_prev(res);
    int posted = 0;
    for (int i = 0; i < data->dp_size; ++i) {
        const int dst = data->inter_node_dst[i];
        if (dst != -1) {
            if (ibv_tools::post_write(res, i * chunk, data->dp_rank * chunk,
                    static_cast<std::size_t>(data->num_tokens_cpu[dst]) * data->data_length_per_token + 4, i) == 0) ++posted;
            if (ibv_tools::post_write(res, data->cross_buffer_size + i * 8,
                    data->cross_buffer_size + data->dp_rank * 8, 8, i) == 0) ++posted;
        }
    }
    res->pending_poll_size = posted;
    std::free(args);
}

// Main ELF 0x43bd8. Combine posts every DP peer, without the dispatch filter.
void rdmaPostPollCombine(void* args) {
    auto* data = static_cast<tecoEPIBArgsV*>(args);
    auto* res = data->res;
    const std::size_t chunk = data->cross_buffer_size / static_cast<std::size_t>(data->dp_size);
    poll_prev(res);
    int posted = 0;
    for (int i = 0; i < data->dp_size; ++i) {
        if (ibv_tools::post_write(res, i * chunk, data->dp_rank * chunk,
                static_cast<std::size_t>(data->num_tokens_cpu[i]) * data->data_length_per_token + 4, i) == 0) ++posted;
        if (ibv_tools::post_write(res, data->cross_buffer_size + i * 8,
                data->cross_buffer_size + data->dp_rank * 8, 8, i) == 0) ++posted;
    }
    res->pending_poll_size = posted;
    std::free(args);
}

// Main ELF 0x43d38. Unlike dispatch, all byte-count entries are posted.
void rdmaPostPollIBFuncCombineData(void* args) {
    auto* data = static_cast<tecoEPIBArgsL*>(args);
    auto* res = data->res;
    poll_prev(res);
    int posted = 0;
    for (int i = 0; i < data->dp_size; ++i) {
        const std::size_t chunk = data->cross_buffer_size / static_cast<std::size_t>(data->dp_size);
        if (ibv_tools::post_write(res, i * chunk, data->dp_rank * chunk,
                static_cast<std::size_t>(data->data_size[i]), i) == 0) ++posted;
    }
    res->pending_poll_size = posted;
    std::free(args);
}
