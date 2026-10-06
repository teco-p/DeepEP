#include "ib_tools.hpp"
#include "exception.hpp"
#include <arpa/inet.h>
#include <endian.h>
#include <sys/time.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

// Original config.cpp owns this map; its loading lifecycle is recovered separately.
extern std::unordered_map<std::string, std::string> envMap;

namespace ibv_tools {
namespace {
bool roce_compat() {
    const char* value = std::getenv("TECOEP_ROCE_COMPAT");
    return value && std::strcmp(value, "1") == 0;
}
// Main ELF 0x3afe0. Only compatibility mode honors the decimal GID selector.
int tcclIBGidSelect() {
    if (!roce_compat()) return 0;
    const char* value = std::getenv("TCCL_IB_GID");
    if (!value) return 0;
    char* end = nullptr;
    const long gid = std::strtol(value, &end, 10);
    return end != value && *end == '\0' && gid >= 0 ? static_cast<int>(gid) : 0;
}
}

// Main ELF 0x3b090.
bool ib_device_check(ibv_device* device) {
    ibv_context* context = ibv_open_device(device);
    if (!context) return false;
    ibv_device_attr device_attr;
    ibv_port_attr port_attr;
    bool result = false;
    if (ibv_query_device(context, &device_attr) == 0 && ibv_query_port(context, 1, &port_attr) == 0 && port_attr.state == 4) {
        result = port_attr.link_layer == 1 || (roce_compat() && port_attr.link_layer == 2);
    }
    if (!result) std::puts("IBV is not OK");
    ibv_close_device(context);
    return result;
}

// Main ELF 0x3bcb0. Preserve original per-card selection and base-0 parsing.
int tcclIBDevSelect(int device_id, int device_count) {
    int selected[8]{};
    const char* value = std::getenv("TCCL_IB_DEV");
    if (value) value = std::getenv("TCCL_IB_DEV");
    else {
        if (!envMap.count("TCCL_IB_DEV")) return 0;
        value = envMap["TCCL_IB_DEV"].c_str();
    }
    if (!value) return 0;
    const int length = static_cast<int>(std::strlen(value));
    int position = 0, slot = 0;
    while (position < length) {
        while (position < length && (value[position] < '0' || value[position] > '9')) ++position;
        int item = static_cast<int>(std::strtol(value + position, nullptr, 0));
        if (item >= device_count) item = 0;
        selected[slot] = item;
        while (position < length && value[position] >= '0' && value[position] <= '9') ++position;
        if (++slot == 8 || position >= length) break;
    }
    return selected[device_id / 4];
}

// Main ELF 0x3b1d8, 0x3b2c0, 0x3b3b0. Numeric masks retain original verbs contract.
int modify_qp_to_init(ibv_qp* qp) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = 1;
    attr.qp_access_flags = roce_compat() ? 7 : 31;
    const int result = ibv_modify_qp(qp, &attr, 0x39);
    if (result) std::puts("failed to modify QP state to INIT");
    return result;
}
int modify_qp_to_rtr(ibvResources* res, ibv_qp* qp, std::uint32_t remote_qpn,
                     std::uint16_t remote_lid, std::uint8_t* remote_gid) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = res->port_attr.active_mtu;
    attr.dest_qp_num = remote_qpn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    std::memcpy(attr.ah_attr.grh.dgid.raw, remote_gid, 16);
    attr.ah_attr.grh.sgid_index = tcclIBGidSelect();
    attr.ah_attr.grh.hop_limit = 1;
    attr.ah_attr.dlid = remote_lid;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.port_num = 1;
    const int result = ibv_modify_qp(qp, &attr, 0x129181);
    if (result) std::puts("failed to modify QP state to RTR");
    return result;
}
int modify_qp_to_rts(ibv_qp* qp) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.max_rd_atomic = 1;
    const int result = ibv_modify_qp(qp, &attr, 0x12e01);
    if (result) std::puts("failed to modify QP state to RTS");
    return result;
}

// Main ELF 0x3b440. Wire fields arrive in network order; GID remains raw bytes.
int connect_qp(ibvResources* res, std::vector<cm_con_data_t>& data, int dp_size) {
    if (static_cast<int>(data.size()) != dp_size) {
        throw EPException("Assertion failed: (int)tmp_con_data.size() == MAX_PEER_NUM",
            "/data/ci_env/slave_loongson_tecoep_py312/workspace/loongson_build_tecoep_py312/usertestdir_3/tecoep/csrc/ib_tools.cpp", 345);
    }
    for (int i = 0; i < dp_size; ++i) {
        cm_con_data_t remote = data[i];
        remote.addr = be64toh(remote.addr);
        remote.rkey = ntohl(remote.rkey);
        remote.qp_num = ntohl(remote.qp_num);
        remote.lid = ntohs(remote.lid);
        res->remote_props[i] = remote;
        int status = modify_qp_to_init(res->qp[i]);
        if (status) return status;
        status = modify_qp_to_rtr(res, res->qp[i], remote.qp_num, remote.lid, remote.gid);
        if (status) return status;
        status = modify_qp_to_rts(res->qp[i]);
        if (status) return status;
    }
    return 0;
}

// Main ELF 0x3b6c0.
int post_write(ibvResources* res, std::size_t local_offset, std::size_t remote_offset,
               std::size_t length, int qp_index) {
    ibv_sge sge{};
    sge.addr = reinterpret_cast<std::uintptr_t>(res->hostAddr) + local_offset;
    sge.length = static_cast<std::uint32_t>(length);
    sge.lkey = res->mr->lkey;
    ibv_send_wr wr{};
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = 2;
    wr.wr.rdma.remote_addr = res->remote_props[qp_index].addr + remote_offset;
    wr.wr.rdma.rkey = res->remote_props[qp_index].rkey;
    ibv_send_wr* bad_wr;
    const int status = ibv_post_send(res->qp[qp_index], &wr, &bad_wr);
    if (status) std::puts("failed to post SR");
    return status;
}

// Main ELF 0x3b7c8. Original returns requested size even after timeout/poll error.
int poll_completion(ibvResources* res, int poll_size, int) {
    timeval time;
    gettimeofday(&time, nullptr);
    const long start_ms = time.tv_sec * 1000 + time.tv_usec / 1000;
    ibv_wc completions[100];
    int completed = 0;
    for (;;) {
        const int status = ibv_poll_cq(res->cq, poll_size, completions);
        gettimeofday(&time, nullptr);
        completed += status;
        if (completed >= poll_size || status < 0) {
            if (status < 0) std::puts("poll CQ failed");
            return poll_size;
        }
        if (static_cast<unsigned long>(time.tv_sec * 1000 + time.tv_usec / 1000 - start_ms) >= 1800000ul) {
            std::printf("completion wasn't found in the CQ after timeout, poll_size:%d\n", poll_size);
            return poll_size;
        }
    }
}

// Main ELF 0x3b950. Unlike failure cleanup, this function does not clear fields.
int resources_destroy(ibvResources* res, int dp_size) {
    int result = 0;
    for (int i = 0; i < dp_size; ++i) {
        if (res->qp[i] && ibv_destroy_qp(res->qp[i])) { std::fputs("failed to destroy QP\n", stderr); result = 1; }
    }
    if (res->cq && ibv_destroy_cq(res->cq)) { std::fputs("failed to destroy CQ\n", stderr); result = 1; }
    if (res->mr && ibv_dereg_mr(res->mr)) { std::fputs("failed to deregister MR\n", stderr); result = 1; }
    if (res->pd && ibv_dealloc_pd(res->pd)) { std::fputs("failed to deallocate PD\n", stderr); result = 1; }
    if (res->ib_ctx && ibv_close_device(res->ib_ctx)) { std::fputs("failed to close device context\n", stderr); return 1; }
    return result;
}

// Main ELF 0x3bb28.
int get_con_data(ibvResources* res, int dp_size) {
    const int gid_index = tcclIBGidSelect();
    res->local_props.resize(dp_size);
    for (int i = 0; i < dp_size; ++i) {
        auto& local = res->local_props[i];
        local.addr = htobe64(reinterpret_cast<std::uintptr_t>(res->hostAddr));
        local.rkey = htonl(res->mr->rkey);
        local.qp_num = htonl(res->qp[i]->qp_num);
        local.lid = htons(res->port_attr.lid);
        std::memset(local.gid, 0, 16);
        if (ibv_query_gid(res->ib_ctx, 1, gid_index, reinterpret_cast<ibv_gid*>(local.gid))) {
            std::printf("can't read sgid of index %d\n", gid_index); return 1;
        }
    }
    return 0;
}

// Main ELF 0x3bfa8. Preserve shared context/PD/MR reuse and original failure cleanup.
int resources_create(ibvResources* res, void* host, void* device, int device_id, int dp_size) {
    if (ibv_fork_init()) std::puts("ibv_fork_init failed !!!");
    int count;
    ibv_device** devices = ibv_get_device_list(&count);
    if (!devices || !count) { std::printf("failed to get IB devices list, num:%d\n", count); goto fail; }
    if (!res->ib_ctx) {
        const int selected = tcclIBDevSelect(device_id, count);
        if (selected >= 0 && selected < count && ib_device_check(devices[selected])) res->ib_ctx = ibv_open_device(devices[selected]);
        if (!res->ib_ctx) { std::printf("device %d failed to open IB device, %s\n", device_id, ibv_get_device_name(devices[selected])); goto fail; }
    }
    {
        ibv_device_attr attr;
        if (ibv_query_device(res->ib_ctx, &attr)) { std::puts("ibv_query_device"); goto fail; }
    }
    if (ibv_query_port(res->ib_ctx, 1, &res->port_attr)) { std::printf("ibv_query_port on port %u failed\n", 1u); goto fail; }
    if (!res->pd) {
        res->pd = ibv_alloc_pd(res->ib_ctx);
        if (!res->pd) { std::puts("ibv_alloc_pd failed"); goto fail; }
    }
    res->hostAddr = host;
    res->deviAddr = device;
    if (!res->mr) {
        const unsigned int flags = roce_compat() ? 7 : 0x10001f;
        res->mr = ibv_reg_mr_iova2(res->pd, res->hostAddr, res->size,
            reinterpret_cast<std::uintptr_t>(res->hostAddr), flags);
        if (!res->mr) { std::printf("ibv_reg_mr failed with mr_flags=0x%x\n", flags); goto fail; }
    }
    res->cq = ibv_create_cq(res->ib_ctx, 256, nullptr, nullptr, 0);
    if (!res->cq) { std::printf("failed to create CQ with %u entries\n", 256u); goto fail; }
    res->qp.resize(dp_size);
    for (int i = 0; i < dp_size; ++i) {
        ibv_qp_init_attr attr{};
        attr.send_cq = res->cq;
        attr.recv_cq = res->cq;
        attr.cap.max_send_wr = 256;
        attr.cap.max_recv_wr = 256;
        attr.cap.max_send_sge = 1;
        attr.cap.max_recv_sge = 1;
        attr.qp_type = IBV_QPT_RC;
        attr.sq_sig_all = 1;
        res->qp[i] = ibv_create_qp(res->pd, &attr);
        if (!res->qp[i]) { std::puts("failed to create QP"); goto fail; }
    }
    return 0;
fail:
    for (auto*& qp : res->qp) { if (qp) { ibv_destroy_qp(qp); qp = nullptr; } }
    if (res->cq) { ibv_destroy_cq(res->cq); res->cq = nullptr; }
    if (res->mr) { ibv_dereg_mr(res->mr); res->mr = nullptr; }
    if (res->pd) { ibv_dealloc_pd(res->pd); res->pd = nullptr; }
    if (res->ib_ctx) { ibv_close_device(res->ib_ctx); res->ib_ctx = nullptr; }
    if (devices) ibv_free_device_list(devices);
    return 1;
}
} // namespace ibv_tools
