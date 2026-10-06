#include "device_info.hpp"
#include "runtime_check.hpp"
#include <algorithm>
#include <iostream>

namespace teco_ep {

// Recovered main ELF 0x192f8, including signed division and SDAA ordinal query.
DeviceInfo::DeviceInfo(int rank_, int num_ranks_, int dp_size_, int ep_size_)
    : rank(rank_), num_ranks(num_ranks_), dp_size(dp_size_), ep_size(ep_size_) {
    ep_sdaa_check(sdaaGetDevice(&device_id), "sdaaGetDevice(&device_id)", "DeviceInfo", 45);
    node_id = rank / 32;
    node_num = num_ranks <= 32 ? 1 : (num_ranks + 31) / 32;
    rank_id_local_card = rank % 4;
    rank_id_local_node = rank % 32;
    card_id_local_node = rank_id_local_node / 4;
    num_ranks_per_node = std::min(num_ranks, 32);
    num_card_per_node = num_ranks_per_node / 4;
    const int ranks_per_dp = num_ranks / dp_size;
    dp_rank = rank / ranks_per_dp;
    card_num_per_dp = ranks_per_dp / 4;
    card_id_per_dp = card_id_local_node % card_num_per_dp;
    for (int i = 0; i < dp_size; ++i) {
        dst_ep_with_dp.push_back(i * card_num_per_dp + card_id_per_dp);
    }
    local_ep_id = dst_ep_with_dp[dp_rank];
}

// Recovered main ELF 0x134e8. Preserve field order and original flushes.
void DeviceInfo::printInfo() const {
    std::cout << "========== DeviceInfo ==========" << std::endl
              << "[Global]" << std::endl
              << "  rank                 = " << rank << std::endl
              << "  num_ranks            = " << num_ranks << std::endl
              << "  dp_size              = " << dp_size << std::endl
              << "  device_id            = " << device_id << std::endl
              << "[Node]" << std::endl
              << "  node_id              = " << node_id << std::endl
              << "  node_num             = " << node_num << std::endl
              << "  num_ranks_per_node   = " << num_ranks_per_node << std::endl
              << "  num_card_per_node    = " << num_card_per_node << std::endl
              << "[Local Rank Mapping]" << std::endl
              << "  rank_id_local_node   = " << rank_id_local_node << std::endl
              << "  rank_id_local_card   = " << rank_id_local_card << std::endl
              << "  card_id_local_node   = " << card_id_local_node << std::endl
              << "[DP / EP Mapping]" << std::endl
              << "  dp_rank              = " << dp_rank << std::endl
              << "  card_num_per_dp      = " << card_num_per_dp << std::endl
              << "  card_id_per_dp       = " << card_id_per_dp << std::endl
              << "  local_ep_id          = " << local_ep_id << std::endl
              << "  dst_ep_with_dp       = [ ";
    for (std::size_t i = 0; i < dst_ep_with_dp.size(); ++i) {
        std::cout << dst_ep_with_dp[i];
        if (i + 1 < dst_ep_with_dp.size()) std::cout << ", ";
    }
    std::cout << " ]" << std::endl << "================================\n" << std::endl;
}

} // namespace teco_ep
