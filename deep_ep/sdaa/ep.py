"""DeepEP v2 dispatch/handle/combine for the SDAA TP-replicated EP16 topology.

The transport is native SDAA IPC/DMA. Counts and inverse routing stay on device.
The four SPA inputs in each card are TP replicas; independent 16-source-rank
input is not this topology and is rejected rather than silently deduplicated.
"""
from dataclasses import dataclass
import torch
import torch.distributed as dist
from .. import _sdaa


class EventOverlap:
    def __init__(self, event=None, extra_tensors=()):
        self.event = event
        self.extra_tensors = extra_tensors

    def current_stream_wait(self, release_handle=False):
        if self.event is not None:
            torch.sdaa.current_stream().wait_event(self.event)
        if release_handle:
            self.event = None
            self.extra_tensors = ()

    wait = current_stream_wait


@dataclass
class EPHandle:
    owner: object
    topk_idx: torch.Tensor
    version: object
    num_experts: int
    num_max_tokens_per_rank: int
    num_recv_tokens: int
    psum_num_recv_tokens_per_scaleup_rank: torch.Tensor
    psum_num_recv_tokens_per_expert: torch.Tensor
    recv_src_metadata: torch.Tensor
    dst_buffer_slot_idx: torch.Tensor
    card_counts: torch.Tensor
    layout: torch.Tensor
    send_counts: torch.Tensor
    original_shape: tuple
    retained: tuple
    do_expand: bool = False
    expert_alignment: int = 1
    num_recv_tokens_per_expert_list: object = None

    def check(self, owner):
        if self.owner is not owner:
            raise ValueError("EPHandle belongs to a different EPBuffer")
        if self.version is not None and self.topk_idx._version != self.version:
            raise RuntimeError("dispatch top-k IDs changed while the handle was in use")


class EPBuffer:
    def __init__(self, group, num_bytes=64 * 1024 * 1024, *,
                 num_max_tokens_per_rank, hidden=5120, num_topk=6,
                 num_experts=384, tp_replica_size=4):
        if (dist.get_world_size(group) != 16 or tp_replica_size != 4
                or hidden != 5120 or num_topk != 6 or num_experts != 384
                or not 0 < num_max_tokens_per_rank <= 16000):
            raise ValueError("SDAA EPBuffer requires TP4-replicated EP16, H5120/K6/E384")
        if num_max_tokens_per_rank * ((hidden + 16) * 2 + num_topk * 8) + 4 > num_bytes // 4:
            raise ValueError("maximum sender rows exceed the native cross-buffer chunk")
        self.group = group
        self.rank = dist.get_rank(group)
        self.dp_rank = self.rank // 4
        self.device = torch.device("sdaa", torch.sdaa.current_device())
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.capacity = num_max_tokens_per_rank * 4
        self.runtime = _sdaa.Buffer(self.rank, 16, 4, 4, num_bytes, False)
        self.runtime.set_transport_token_capacity(num_max_tokens_per_rank)
        def gather(value):
            result = [None] * 16
            dist.all_gather_object(result, value, group)
            return result
        self.runtime.sync(gather(self.runtime.get_dispatch_ipc_handle()),
                          gather(self.runtime.get_dispatch_props()),
                          gather(self.runtime.get_combine_ipc_handle()),
                          gather(self.runtime.get_combine_props()))
        if not self.runtime.is_available():
            raise RuntimeError("native SDAA transport initialization failed")
        self.routing_status = torch.zeros(4, dtype=torch.int64, device=self.device)
        self._source_rows = torch.arange(num_max_tokens_per_rank, dtype=torch.int16).to(self.device)

    @staticmethod
    def capture():
        event = torch.sdaa.Event()
        event.record(torch.sdaa.current_stream())
        return event

    def get_native_status(self):
        return self.runtime.get_epoch_status(), self.routing_status

    def dispatch(self, x, topk_idx, topk_weights, *, num_experts=384,
                 num_max_tokens_per_rank=None, expert_alignment=1,
                 do_cpu_sync=False, do_expand=False, previous_event=None,
                 async_with_compute_stream=False, allocate_on_comm_stream=False,
                 handle=None, defer_epilogue=False, **unsupported):
        if (unsupported or do_cpu_sync or do_expand or expert_alignment != 1
                or async_with_compute_stream or allocate_on_comm_stream
                or defer_epilogue or handle is not None or num_experts != 384
                or num_max_tokens_per_rank not in (None, self.num_max_tokens_per_rank)):
            raise ValueError("unsupported SDAA dispatch mode; no CPU count sync or cached routing")
        if previous_event is not None:
            previous_event.current_stream_wait()
        n = x.shape[0]
        if (x.shape != (n, 5120) or x.dtype != torch.float16
                or x.device != self.device or n > self.num_max_tokens_per_rank
                or topk_idx.shape != (n, 6) or topk_idx.dtype not in (torch.int32, torch.int64)
                or topk_weights.shape != (n, 6) or topk_weights.dtype != torch.float32
                or topk_idx.device != x.device or topk_weights.device != x.device
                or not x.is_contiguous() or not topk_idx.is_contiguous()
                or not topk_weights.is_contiguous()):
            raise ValueError("SDAA normalized FP16/FP32 route input differs from the loaded topology")
        ids = torch.empty((n, 6), dtype=torch.int32, device=x.device)
        _sdaa.cast_indices(topk_idx, ids, self.routing_status, 384)
        layout = torch.zeros((n + 1, 4), dtype=torch.int32, device=x.device)
        send_counts = torch.empty(4, dtype=torch.int32, device=x.device)
        wire_x = torch.zeros((n, 5136), dtype=torch.float16, device=x.device)
        wire_x[:, :5120].copy_(x)
        # Opaque metadata bytes: original row, not a numeric FP16 conversion.
        wire_x.view(torch.int16)[:, 5120].copy_(self._source_rows[:n])
        self.runtime.get_dispatch_layout(ids, topk_weights, wire_x, 96, 384, 4, 4, layout, send_counts)
        self.runtime.dispatch_data(ids, topk_weights, wire_x, 4, 4, send_counts)
        card_counts = torch.empty(5, dtype=torch.int32, device=x.device)
        card_ids = torch.full((self.capacity, 6), -1, dtype=torch.int32, device=x.device)
        card_weights = torch.zeros((self.capacity, 6), dtype=torch.float32, device=x.device)
        card_x = torch.zeros((self.capacity, 5136), dtype=torch.float16, device=x.device)
        self.runtime.gather_dispatch_data(self.dp_rank, 4, card_counts, card_ids, card_weights, card_x)
        recv_ids = torch.full_like(card_ids, -1)
        recv_weights = torch.zeros_like(card_weights)
        recv_x = torch.zeros((self.capacity, 5120), dtype=x.dtype, device=x.device)
        rank_prefix = torch.zeros(16, dtype=torch.int32, device=x.device)
        expert_prefix = torch.zeros(24, dtype=torch.int32, device=x.device)
        inverse = torch.full((self.capacity,), -1, dtype=torch.int32, device=x.device)
        src = torch.full((self.capacity, 2), -1, dtype=torch.int32, device=x.device)
        _sdaa.compact_dispatch(card_x, card_ids, card_weights, card_counts,
            recv_x, recv_ids, recv_weights, rank_prefix, expert_prefix,
            inverse, src, self.rank, self.routing_status)
        try:
            version = topk_idx._version
        except RuntimeError:
            version = None
        retained = (x, wire_x, ids, topk_weights, card_x, card_ids, card_weights, recv_x,
                    recv_ids, recv_weights, rank_prefix, expert_prefix, inverse, src,
                    card_counts, layout, send_counts, self.routing_status)
        handle = EPHandle(self, topk_idx, version, num_experts,
            self.num_max_tokens_per_rank, self.capacity, rank_prefix, expert_prefix,
            src, inverse, card_counts, layout, send_counts, tuple(x.shape), retained)
        return recv_x, recv_ids, recv_weights, handle, EventOverlap(self.capture(), retained)

    def combine(self, x, handle, topk_weights=None, *, previous_event=None,
                async_with_compute_stream=False, allocate_on_comm_stream=False,
                defer_epilogue=False, **unsupported):
        handle.check(self)
        if (topk_weights is not None or unsupported or async_with_compute_stream
                or allocate_on_comm_stream or defer_epilogue
                or x.shape != (self.capacity, 5120) or x.dtype != torch.float16
                or x.device != self.device or not x.is_contiguous()):
            raise ValueError("SDAA combine requires preweighted FP16 expert outputs and the original handle")
        if previous_event is not None:
            previous_event.current_stream_wait()
        # Each SPA scatters only its expert contribution back to the card layout.
        # The mature native card reduction sums these contributions exactly once.
        card_output = torch.zeros_like(x)
        _sdaa.scatter_combine(x, handle.dst_buffer_slot_idx, card_output, self.routing_status)
        self.runtime.get_combine_layout(card_output, handle.card_counts, 4, 4)
        self.runtime.combine_data(card_output, 4, 4, handle.card_counts)
        output = torch.zeros(handle.original_shape, dtype=x.dtype, device=x.device)
        self.runtime.gather_combine_data(self.dp_rank, 4, 4,
            handle.layout, handle.send_counts, output)
        retained = (handle, x, card_output, output)
        return output, None, EventOverlap(self.capture(), retained)
