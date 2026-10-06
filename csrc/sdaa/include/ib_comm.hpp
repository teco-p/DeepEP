#pragma once
#include "ib_tools.hpp"
#include <cstddef>

// Layouts from the original ops.hpp DWARF. The trailing fields are present in
// the original types even where the generated host code does not use them.
struct tecoEPIbArgs {
    ibv_tools::ibvResources* res;
    std::size_t offset, dst_offset, data_length_per_token, data_size;
    int* num_tokens_ptr;
    int peer;
    bool if_poll;
    int poll_size;
};
struct tecoEPIBArgsV {
    ibv_tools::ibvResources* res;
    std::size_t cross_buffer_size, data_length_per_token;
    int* inter_node_dst;
    int* num_tokens_cpu;
    int dp_rank, dp_size, rank;
};
struct tecoEPIBArgsL {
    ibv_tools::ibvResources* res;
    std::size_t cross_buffer_size;
    int* inter_node_dst;
    int* data_size;
    int dp_rank, dp_size, rank;
};
static_assert(sizeof(tecoEPIbArgs) == 64);
static_assert(offsetof(tecoEPIbArgs, if_poll) == 52);
static_assert(sizeof(tecoEPIBArgsV) == 56);
static_assert(sizeof(tecoEPIBArgsL) == 48);

void rdmaPostIbFunc(void* args);
void rdmaPostPollIBFunc(void* args);
void rdmaPostPollIBFuncData(void* args);
void rdmaPostPollCombine(void* args);
void rdmaPostPollIBFuncCombineData(void* args);
