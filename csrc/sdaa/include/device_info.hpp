#pragma once

#include <vector>

namespace teco_ep {

// Field order is preserved from g447683d teco_ep.hpp:36 DWARF.
struct DeviceInfo {
    int rank;
    int num_ranks;
    int dp_size;
    int ep_size;
    int device_id;
    int node_id;
    int node_num;
    int rank_id_local_card;
    int rank_id_local_node;
    int card_id_local_node;
    int num_ranks_per_node;
    int num_card_per_node;
    int dp_rank;
    int card_num_per_dp;
    int card_id_per_dp;
    int local_ep_id;
    std::vector<int> dst_ep_with_dp;

    DeviceInfo(int rank, int num_ranks, int dp_size, int ep_size);
    void printInfo() const;
};

} // namespace teco_ep
