#include "buffer.hpp"
#include "exception.hpp"
#include "ib_comm.hpp"
#include "runtime_check.hpp"
#include "epoch.hpp"
#include <torch_sdaa/sdaa_extension.h>
#include <sdaa_experimental_api.h>
#include <cstdlib>
#include <cstdio>

namespace teco_ep {
namespace {
constexpr const char* original_file = "/data/ci_env/slave_loongson_tecoep_py312/workspace/loongson_build_tecoep_py312/usertestdir_3/tecoep/csrc/teco_ep.cpp";
char* bytes(void* p) { return static_cast<char*>(p); }
void count_check(bool condition, const char* reason, int line) {
    if (!condition) throw EPException(reason, original_file, line);
}
}

// Main ELF 0x1b610. Original low-latency branch is a tail call, not the
// spurious series of GOT calls shown by the decompiler at that branch.
void Buffer::dispatch_data(const at::Tensor& topk_idx, const at::Tensor& topk_weight,
    const at::Tensor& hidden_states, int ep_size_, int dp_size_, const at::Tensor& num_tokens_per_card,
    std::int64_t transport_token_bound) {
    if (low_latency_mode) {
        dispatch_data_lowlatency(topk_idx, topk_weight, hidden_states, ep_size_, dp_size_, num_tokens_per_card);
        return;
    }
    count_check(ep_size_ == num_tokens_per_card.size(0),
        "ep_size == num_tokens_per_card.size(0)failed, num_tokens_per_card.size(0) != ep_size", 674);
    count_check(dp_size_ == dp_size && dp_size_ > 0, "dispatch DP size differs from Buffer", 674);
    count_check(transport_token_bound >= hidden_states.size(0),
        "transport bound must cover this source's rows", 674);
    const std::size_t token_bytes_bound = hidden_states.element_size() * hidden_states.size(1)
        + (topk_idx.element_size() + topk_weight.element_size()) * topk_idx.size(1);
    const std::size_t chunk_bound = cross_buffer_size / static_cast<std::size_t>(dp_size_);
    count_check(token_bytes_bound > 0 && chunk_bound >= 4 && static_cast<std::size_t>(transport_token_bound) <=
        (chunk_bound - 4) / token_bytes_bound, "dispatch shape bound exceeds transport chunk", 674);
    const std::size_t copy_bytes = static_cast<std::size_t>(transport_token_bound) * token_bytes_bound + 4;
    auto stream = torch::sdaa::getCurrentSDAAStream(-1);
    const int card = device_info.card_id_local_node;
    if (device_info.rank_id_local_card == 0) {
        char* send = bytes(dispatch.send_buffer_ptrs[card]);
        const std::size_t chunk = cross_buffer_size / static_cast<std::size_t>(dp_size_);
        const std::int64_t token_bytes = hidden_states.element_size() * hidden_states.size(1)
            + (topk_idx.element_size() + topk_weight.element_size()) * topk_idx.size(1);
        ep_sdaa_check(sdaaStreamWriteValue64(stream, &dispatch.send_barrier_ptrs[card][dp_rank], 1, SDAA_STREAM_WRITE_VALUE_ADD),
            "sdaaStreamWriteValue64( stream, &dispatch.send_barrier_ptrs[device_info.card_id_local_node][dp_rank], 1, SDAA_STREAM_WRITE_VALUE_ADD)", "dispatch_data", 689);
        ep_sdaa_check(sdaaMemcpyAsync(&dispatch.recv_barrier_ptrs[card][dp_rank], &dispatch.send_barrier_ptrs[card][dp_rank], sizeof(std::uint64_t), sdaaMemcpyDeviceToDevice, stream),
            "sdaaMemcpyAsync( &dispatch.recv_barrier_ptrs[device_info.card_id_local_node][dp_rank], &dispatch.send_barrier_ptrs[device_info.card_id_local_node][dp_rank], sizeof(uint64_t), sdaaMemcpyDeviceToDevice, stream)", "dispatch_data", 695);
        int remote_peers = 0;
        for (int i = 1; i < dp_size_; ++i) {
            const int next = (dp_rank + i) % dp_size_;
            const int dst = device_info.dst_ep_with_dp[next];
            if (is_same_node(device_info.local_ep_id, dst)) {
                // Headers retain actual counts; capture records a stable shape bound.
                ep_sdaa_check(sdaaIpcMemcpyPeertoPeerAsync(bytes(dispatch.recv_buffer_ptrs[dst % device_info.num_card_per_node]) + dp_rank * chunk,
                    send + next * chunk, copy_bytes, SDAA_IPC_P2P_RO_OPEN, stream),
                    "sdaaIpcMemcpyPeertoPeerAsync(fixed dispatch bound)", "dispatch_data", 721);
                ep_sdaa_check(sdaaStreamWriteValue64(stream, &dispatch.send_barrier_ptrs[card][next], 1, SDAA_STREAM_WRITE_VALUE_ADD),
                    "sdaaStreamWriteValue64( stream, &dispatch .send_barrier_ptrs[device_info.card_id_local_node][dp_next], 1, SDAA_STREAM_WRITE_VALUE_ADD)", "dispatch_data", 727);
                ep_sdaa_check(sdaaIpcMemcpyPeertoPeerAsync(&dispatch.recv_barrier_ptrs[dst % device_info.num_card_per_node][dp_rank],
                    &dispatch.send_barrier_ptrs[card][next], sizeof(std::uint64_t), SDAA_IPC_P2P_RO_OPEN, stream),
                    "sdaaIpcMemcpyPeertoPeerAsync( &dispatch.recv_barrier_ptrs[device_info.dst_ep_with_dp[dp_next] % device_info.num_card_per_node][dp_rank], &dispatch .send_barrier_ptrs[device_info.card_id_local_node][dp_next], sizeof(uint64_t), SDAA_IPC_P2P_RO_OPEN, stream)", "dispatch_data", 733);
            } else {
                ++remote_peers;
                ep_sdaa_check(sdaaStreamWriteValue64(stream, &dispatch.send_barrier_ptrs[card][next], 1, SDAA_STREAM_WRITE_VALUE_ADD),
                    "sdaaStreamWriteValue64( stream, &dispatch .send_barrier_ptrs[device_info.card_id_local_node][dp_next], 1, SDAA_STREAM_WRITE_VALUE_ADD)", "dispatch_data", 743);
            }
        }
        if (device_info.node_num > 0 && remote_peers > 0) {
            dispatch_num_tokens_cpu.copy_(num_tokens_per_card, true);
            auto* args = static_cast<tecoEPIBArgsV*>(std::malloc(sizeof(tecoEPIBArgsV)));
            if (!args) std::puts("ib launch host mem malloc failed");
            else {
                args->res = &dispatch.ibv_send;
                args->cross_buffer_size = cross_buffer_size;
                args->data_length_per_token = token_bytes;
                args->inter_node_dst = inter_node_dst;
                args->num_tokens_cpu = dispatch_num_tokens_cpu.data_ptr<int>();
                args->dp_rank = dp_rank;
                args->dp_size = dp_size_;
                ep_sdaa_check(sdaaLaunchHostFunc(stream, rdmaPostPollIBFunc, args),
                    "sdaaLaunchHostFunc(stream, rdmaPostPollIBFunc, (args_))", "dispatch_data", 754);
            }
        }
    }
    if (tecoep::kernels::wait_receive_epoch(stream, dispatch.recv_barrier_ptrs[card],
        reinterpret_cast<unsigned long*>(dispatch_expected_epoch_.data_ptr<std::int64_t>()),
        dp_rank, dp_size_, 1, epoch_status_.data_ptr<std::int64_t>()) != EP_RETURN_SUCCESS)
        throw EPException("dispatch epoch launch failed", original_file, 767);
}

// Main ELF 0x1bdc8. Receive waits belong to gather_combine_data, not this call.
void Buffer::combine_data(const at::Tensor& hidden_states, int ep_size_, int dp_size_,
    const at::Tensor& recv_num_tokens_per_dp, std::int64_t transport_token_bound) {
    if (low_latency_mode) {
        combine_data_lowlatency(hidden_states, ep_size_, dp_size_, recv_num_tokens_per_dp);
        return;
    }
    count_check(dp_size_ + 1 == recv_num_tokens_per_dp.size(0),
        "dp_size + 1 == recv_num_tokens_per_dp.size(0)failed, recv_num_tokens_per_dp.size(0) != dp_size", 1025);
    count_check(dp_size_ == dp_size && dp_size_ > 0 && transport_token_bound >= 0,
        "combine requires matching preceding dispatch", 1025);
    const std::size_t row_bytes = hidden_states.element_size() * hidden_states.size(1);
    const std::size_t chunk_bound = cross_buffer_size / static_cast<std::size_t>(dp_size_);
    count_check(row_bytes > 0 && chunk_bound >= 4 && static_cast<std::size_t>(transport_token_bound) <=
        (chunk_bound - 4) / row_bytes, "combine shape bound exceeds transport chunk", 1025);
    const std::size_t copy_bytes = static_cast<std::size_t>(transport_token_bound) * row_bytes + 4;
    auto stream = torch::sdaa::getCurrentSDAAStream(-1);
    if (device_info.rank_id_local_card != 0) return;
    const int card = device_info.card_id_local_node;
    char* send = bytes(combine.send_buffer_ptrs[card]);
    const std::size_t chunk = cross_buffer_size / static_cast<std::size_t>(dp_size_);
    const std::int64_t token_bytes = hidden_states.element_size() * hidden_states.size(1);
    ep_sdaa_check(sdaaStreamWriteValue64(stream, &combine.send_barrier_ptrs[card][dp_rank], 1, SDAA_STREAM_WRITE_VALUE_ADD),
        "sdaaStreamWriteValue64( stream, &combine.send_barrier_ptrs[device_info.card_id_local_node][dp_rank], 1, SDAA_STREAM_WRITE_VALUE_ADD)", "combine_data", 1039);
    ep_sdaa_check(sdaaMemcpyAsync(&combine.recv_barrier_ptrs[card][dp_rank], &combine.send_barrier_ptrs[card][dp_rank], sizeof(std::uint64_t), sdaaMemcpyDeviceToDevice, stream),
        "sdaaMemcpyAsync( &combine.recv_barrier_ptrs[device_info.card_id_local_node][dp_rank], &combine.send_barrier_ptrs[device_info.card_id_local_node][dp_rank], sizeof(uint64_t), sdaaMemcpyDeviceToDevice, stream)", "combine_data", 1045);
    int remote_peers = 0;
    for (int i = 1; i < dp_size_; ++i) {
        const int next = (dp_rank + i) % dp_size_;
        const int dst = device_info.dst_ep_with_dp[next];
        if (is_same_node(device_info.local_ep_id, dst)) {
            ep_sdaa_check(sdaaIpcMemcpyPeertoPeerAsync(bytes(combine.recv_buffer_ptrs[dst % device_info.num_card_per_node]) + dp_rank * chunk,
                send + next * chunk, copy_bytes, SDAA_IPC_P2P_RO_OPEN, stream),
                "sdaaIpcMemcpyPeertoPeerAsync(fixed combine bound)", "combine_data", 1071);
            ep_sdaa_check(sdaaStreamWriteValue64(stream, &combine.send_barrier_ptrs[card][next], 1, SDAA_STREAM_WRITE_VALUE_ADD),
                "sdaaStreamWriteValue64( stream, &combine.send_barrier_ptrs[device_info.card_id_local_node][dp_next], 1, SDAA_STREAM_WRITE_VALUE_ADD)", "combine_data", 1075);
            ep_sdaa_check(sdaaIpcMemcpyPeertoPeerAsync(&combine.recv_barrier_ptrs[dst % device_info.num_card_per_node][dp_rank],
                &combine.send_barrier_ptrs[card][next], sizeof(std::uint64_t), SDAA_IPC_P2P_RO_OPEN, stream),
                "sdaaIpcMemcpyPeertoPeerAsync( &combine.recv_barrier_ptrs[device_info.dst_ep_with_dp[dp_next] % device_info.num_card_per_node][dp_rank], &combine.send_barrier_ptrs[device_info.card_id_local_node][dp_next], sizeof(uint64_t), SDAA_IPC_P2P_RO_OPEN, stream)", "combine_data", 1081);
        } else {
            ++remote_peers;
            ep_sdaa_check(sdaaStreamWriteValue64(stream, &combine.send_barrier_ptrs[card][next], 1, SDAA_STREAM_WRITE_VALUE_ADD),
                "sdaaStreamWriteValue64( stream, &combine.send_barrier_ptrs[device_info.card_id_local_node][dp_next], 1, SDAA_STREAM_WRITE_VALUE_ADD)", "combine_data", 1091);
        }
    }
    if (device_info.node_num < 1 || remote_peers < 1) return;
    combine_num_tokens_cpu.copy_(recv_num_tokens_per_dp, true);
    auto* args = static_cast<tecoEPIBArgsV*>(std::malloc(sizeof(tecoEPIBArgsV)));
    if (!args) { std::puts("ib launch host mem malloc failed"); return; }
    args->res = &combine.ibv_send;
    args->cross_buffer_size = cross_buffer_size;
    args->data_length_per_token = token_bytes;
    args->inter_node_dst = inter_node_dst;
    args->num_tokens_cpu = combine_num_tokens_cpu.data_ptr<int>();
    args->dp_rank = dp_rank;
    args->dp_size = dp_size_;
    ep_sdaa_check(sdaaLaunchHostFunc(stream, rdmaPostPollCombine, args),
        "sdaaLaunchHostFunc(stream, rdmaPostPollCombine, (args_))", "combine_data", 1102);
}

// Main ELF 0x13b98. Dynamic-size copies read the packed header directly from
// the source chunk. Low-latency data has 96-byte payloads in 128-byte records.
void Buffer::dispatch_data_lowlatency(const at::Tensor& topk_idx, const at::Tensor& topk_weight,
    const at::Tensor& hidden_states, int ep_size_, int dp_size_, const at::Tensor& num_tokens_per_card) {
    count_check(ep_size_ == num_tokens_per_card.size(0),
        "ep_size == num_tokens_per_card.size(0)failed, num_tokens_per_card.size(0) != ep_size", 789);
    auto stream = torch::sdaa::getCurrentSDAAStream(-1);
    if (device_info.rank_id_local_card != 0 || dp_size_ < 2) return;
    const int card = device_info.card_id_local_node;
    const std::size_t chunk = cross_buffer_size / static_cast<std::size_t>(dp_size_);
    char* send = bytes(dispatch.send_buffer_ptrs[card]);
    int remote_peers = 0;
    for (int i = 1; i < dp_size_; ++i) {
        const int next = (dp_rank + i) % dp_size_;
        const int dst = device_info.dst_ep_with_dp[next];
        if (is_same_node(device_info.local_ep_id, dst)) {
            char* src = send + next * chunk;
            ep_sdaa_check(sdaaIpcMemcpyPeertoPeerAsyncDynamicSize(bytes(dispatch.recv_buffer_ptrs[dst % device_info.num_card_per_node]) + dp_rank * chunk,
                src, src, cross_buffer_size, stream),
                "sdaaIpcMemcpyPeertoPeerAsyncDynamicSize( dst_ptr, src_ptr, src_ptr, cross_buffer_size, stream)", "dispatch_data_lowlatency", 815);
        } else ++remote_peers;
    }
    if (device_info.node_num < 1 || remote_peers < 1) return;
    dispatch_num_tokens_cpu.copy_(num_tokens_per_card, true);
    int* sizes = dispatch_num_tokens_cpu.data_ptr<int>();
    for (int i = 0; i < device_info.ep_size; ++i) {
        const int count = sizes[i];
        const auto idx_bytes = static_cast<std::int64_t>(count) * topk_idx.element_size() * topk_idx.size(1);
        const auto weight_bytes = static_cast<std::int64_t>(count) * topk_weight.element_size() * topk_weight.size(1);
        const auto hidden_bytes = static_cast<std::int64_t>(hidden_states.element_size() * hidden_states.size(1));
        sizes[i] = (static_cast<int>((hidden_bytes + 95) / 96) * count
            + static_cast<int>((weight_bytes + 95) / 96) + static_cast<int>((idx_bytes + 95) / 96) + 1) * 128;
    }
    auto* args = static_cast<tecoEPIBArgsL*>(std::malloc(sizeof(tecoEPIBArgsL)));
    if (!args) { std::puts("ib launch host mem malloc failed"); return; }
    args->res = &dispatch.ibv_send;
    args->cross_buffer_size = cross_buffer_size;
    args->inter_node_dst = inter_node_dst;
    args->data_size = sizes;
    args->dp_rank = dp_rank;
    args->dp_size = dp_size_;
    ep_sdaa_check(sdaaLaunchHostFunc(stream, rdmaPostPollIBFuncData, args),
        "sdaaLaunchHostFunc(stream, rdmaPostPollIBFuncData, (args_))", "dispatch_data_lowlatency", 832);
}

// Main ELF 0x14260. The total entry at index DP is copied but not converted;
// instructions at 0x14450/0x144c8 bound this loop by DP, not DP+1.
void Buffer::combine_data_lowlatency(const at::Tensor& hidden_states, int, int dp_size_,
    const at::Tensor& recv_num_tokens_per_dp) {
    count_check(dp_size_ + 1 == recv_num_tokens_per_dp.size(0),
        "dp_size + 1 == recv_num_tokens_per_dp.size(0)failed, recv_num_tokens_per_dp.size(0) != dp_size", 958);
    auto stream = torch::sdaa::getCurrentSDAAStream(-1);
    if (device_info.rank_id_local_card != 0 || dp_size_ < 2) return;
    const int card = device_info.card_id_local_node;
    const std::size_t chunk = cross_buffer_size / static_cast<std::size_t>(dp_size_);
    char* send = bytes(combine.send_buffer_ptrs[card]);
    int remote_peers = 0;
    for (int i = 1; i < dp_size_; ++i) {
        const int next = (dp_rank + i) % dp_size_;
        const int dst = device_info.dst_ep_with_dp[next];
        if (is_same_node(device_info.local_ep_id, dst)) {
            char* src = send + next * chunk;
            ep_sdaa_check(sdaaIpcMemcpyPeertoPeerAsyncDynamicSize(bytes(combine.recv_buffer_ptrs[dst % device_info.num_card_per_node]) + dp_rank * chunk,
                src, src, cross_buffer_size, stream),
                "sdaaIpcMemcpyPeertoPeerAsyncDynamicSize( dst_ptr, src_ptr, src_ptr, cross_buffer_size, stream)", "combine_data_lowlatency", 986);
        } else ++remote_peers;
    }
    if (device_info.node_num < 1 || remote_peers < 1) return;
    combine_num_tokens_cpu.copy_(recv_num_tokens_per_dp, true);
    int* sizes = combine_num_tokens_cpu.data_ptr<int>();
    for (int i = 0; i < dp_size_; ++i) {
        const auto row_bytes = static_cast<std::int64_t>(hidden_states.element_size() * hidden_states.size(1));
        sizes[i] = (static_cast<int>((row_bytes + 95) / 96) * sizes[i] + 1) * 128;
    }
    auto* args = static_cast<tecoEPIBArgsL*>(std::malloc(sizeof(tecoEPIBArgsL)));
    if (!args) { std::puts("ib launch host mem malloc failed"); return; }
    args->res = &combine.ibv_send;
    args->cross_buffer_size = cross_buffer_size;
    args->inter_node_dst = inter_node_dst;
    args->data_size = sizes;
    args->dp_rank = dp_rank;
    args->dp_size = dp_size_;
    ep_sdaa_check(sdaaLaunchHostFunc(stream, rdmaPostPollIBFuncCombineData, args),
        "sdaaLaunchHostFunc(stream, rdmaPostPollIBFuncCombineData, (args_))", "combine_data_lowlatency", 1002);
}

// Main ELF 0x13128. Copies only the four-byte header at each chunk start.
void Buffer::debug_dump_combine_headers(at::Tensor out_headers, bool from_recv) {
    if (!out_headers.device().is_cpu()) c10::detail::torchCheckFail("debug_dump_combine_headers", original_file, 1152, "out_headers must be on CPU");
    if (out_headers.scalar_type() != at::kInt) c10::detail::torchCheckFail("debug_dump_combine_headers", original_file, 1153, "out_headers must be int32");
    if (out_headers.numel() < device_info.dp_size) c10::detail::torchCheckFail("debug_dump_combine_headers", original_file, 1156, "out_headers too small");
    int* out = out_headers.data_ptr<int>();
    const int card = device_info.card_id_local_node;
    char* src = bytes(from_recv ? combine.recv_buffer_ptrs[card] : combine.send_buffer_ptrs[card]);
    const std::size_t chunk = cross_buffer_size / static_cast<std::size_t>(device_info.dp_size);
    for (int i = 0; i < device_info.dp_size; ++i) {
        ep_sdaa_check(sdaaMemcpy(out + i, src + i * chunk, sizeof(std::int32_t), sdaaMemcpyDeviceToHost),
            "sdaaMemcpy(&out[dp_id], src, sizeof(int32_t), sdaaMemcpyDeviceToHost)", "debug_dump_combine_headers", 1168);
    }
}
} // namespace teco_ep
