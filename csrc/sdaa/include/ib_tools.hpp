#pragma once
#include <infiniband/verbs.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ibv_tools {
#pragma pack(push, 1)
struct cm_con_data_t {
    std::uint64_t addr;
    std::uint32_t rkey;
    std::uint32_t qp_num;
    std::uint16_t lid;
    std::uint8_t gid[16];
};
#pragma pack(pop)
static_assert(sizeof(cm_con_data_t) == 34);
struct ibvResources {
    ibv_device_attr device_attr;
    ibv_port_attr port_attr;
    std::vector<cm_con_data_t> remote_props;
    std::vector<cm_con_data_t> local_props;
    ibv_context* ib_ctx;
    ibv_pd* pd;
    ibv_cq* cq;
    std::vector<ibv_qp*> qp;
    ibv_mr* mr;
    void* hostAddr;
    void* deviAddr;
    std::size_t size;
    int pending_poll_size;
};
static_assert(sizeof(ibvResources) == 424);
static_assert(offsetof(ibvResources, remote_props) == 288);
static_assert(offsetof(ibvResources, pending_poll_size) == 416);
bool ib_device_check(ibv_device* device);
int tcclIBDevSelect(int device_id, int device_count);
int modify_qp_to_init(ibv_qp* qp);
int modify_qp_to_rtr(ibvResources* res, ibv_qp* qp, std::uint32_t remote_qpn,
                     std::uint16_t remote_lid, std::uint8_t* remote_gid);
int modify_qp_to_rts(ibv_qp* qp);
int connect_qp(ibvResources* res, std::vector<cm_con_data_t>& data, int dp_size);
int post_write(ibvResources* res, std::size_t local_offset, std::size_t remote_offset,
               std::size_t length, int qp_index);
int poll_completion(ibvResources* res, int poll_size, int qp_index);
int resources_destroy(ibvResources* res, int dp_size);
int resources_create(ibvResources* res, void* host, void* device, int device_id, int dp_size);
int get_con_data(ibvResources* res, int dp_size);
} // namespace ibv_tools
