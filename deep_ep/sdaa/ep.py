"""DeepEP v2 dispatch/handle/combine for SDAA TP4-replicated EP16/EP32.

The transport is native SDAA IPC/DMA. Counts and inverse routing stay on device.
The four SPA inputs in each card are TP replicas; independent source-rank
input is not this topology and is rejected rather than silently deduplicated.
"""
from dataclasses import dataclass
import torch
import torch.distributed as dist
from .. import _sdaa


class EventOverlap:
    """Synchronous-API same-stream ordering and tensor lifetime, not GPU completion.

    Like DeepEP's async_finish=False return, no event is manufactured. SDAA
    event recording is not supported inside capture; cross-stream use is denied.
    """
    def __init__(self, stream, extra_tensors=()):
        self.event = None
        self.stream = stream
        self.extra_tensors = extra_tensors

    def current_stream_wait(self, release_handle=False):
        if torch.sdaa.current_stream() != self.stream:
            raise ValueError("SDAA synchronous EP overlap cannot transfer a cross-stream dependency")
        if release_handle:
            self.extra_tensors = ()

    wait = current_stream_wait


@dataclass
class EPHandle:
    owner: object
    topk_idx: torch.Tensor
    version: object
    num_experts: int
    num_max_tokens_per_rank: int
    num_recv_tokens: torch.Tensor
    psum_num_recv_tokens_per_scaleup_rank: torch.Tensor
    psum_num_recv_tokens_per_expert: torch.Tensor
    recv_src_metadata: torch.Tensor
    dst_buffer_slot_idx: torch.Tensor
    card_counts: torch.Tensor
    layout: torch.Tensor
    send_counts: torch.Tensor
    original_shape: tuple
    retained: tuple
    num_unaligned_recv_tokens_per_expert: torch.Tensor
    num_padded_recv_tokens_per_expert: torch.Tensor
    num_expanded_tokens: torch.Tensor
    stream: object
    do_expand: bool = True
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
                 num_experts=384, tp_replica_size=4, expert_alignment=1):
        world_size = dist.get_world_size(group)
        if (world_size not in (16, 32) or tp_replica_size != 4
                or hidden != 5120 or num_topk != 6 or num_experts != 384
                or not 0 < num_max_tokens_per_rank <= 16000
                or type(expert_alignment) is not int or expert_alignment <= 0
                or expert_alignment & (expert_alignment - 1)):
            raise ValueError("SDAA EPBuffer requires TP4-replicated EP16/EP32, H5120/K6/E384")
        self.world_size = world_size
        self.dp_size = world_size // 4
        self.local_experts = num_experts // world_size
        if num_max_tokens_per_rank * ((hidden + 16) * 2 + num_topk * 8) + 4 > num_bytes // self.dp_size:
            raise ValueError("maximum sender rows exceed the native cross-buffer chunk")
        self.group = group
        self.rank = dist.get_rank(group)
        self.dp_rank = self.rank // 4
        self.device = torch.device("sdaa", torch.sdaa.current_device())
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.capacity = num_max_tokens_per_rank * self.dp_size
        self.expert_alignment = expert_alignment
        self.expanded_capacity = self.capacity * 6 + self.local_experts * (expert_alignment - 1)
        self.runtime = _sdaa.Buffer(self.rank, world_size, self.dp_size, self.dp_size, num_bytes, False)
        self.runtime.set_transport_token_capacity(num_max_tokens_per_rank)
        def gather(value):
            result = [None] * world_size
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
        # Fixed addresses are shared by eager and captured execution. Only native
        # counts/slot metadata and valid rows are rewritten for each dispatch.
        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=self.device)
        self._ids = zeros((num_max_tokens_per_rank, 6), torch.int32)
        self._wire_x = zeros((num_max_tokens_per_rank, 5136), torch.float16)
        self._wire_x.view(torch.int16)[:, 5120].copy_(self._source_rows)
        self._layout = zeros((num_max_tokens_per_rank + 1, self.dp_size), torch.int32)
        self._send_counts = zeros(self.dp_size, torch.int32)
        self._card_counts = zeros(self.dp_size + 1, torch.int32)
        self._card_ids = zeros((self.capacity, 6), torch.int32)
        self._card_weights = zeros((self.capacity, 6), torch.float32)
        self._card_x = zeros((self.capacity, 5136), torch.float16)
        self._expanded_x = zeros((self.expanded_capacity, 5120), torch.float16)
        self._expanded_weights = zeros(self.expanded_capacity, torch.float32)
        self._rank_prefix = zeros(world_size, torch.int32)
        self._expert_prefix = zeros(self.local_experts, torch.int32)
        self._expert_counts = zeros(self.local_experts, torch.int32)
        self._padded_counts = zeros(self.local_experts, torch.int32)
        self._route_slots = zeros((self.capacity, 6), torch.int32)
        self._source = zeros((self.capacity, 2), torch.int32)
        self._card_output = zeros((self.capacity, 5120), torch.float16)
        self._output = zeros((num_max_tokens_per_rank, 5120), torch.float16)
        self._pending = None

    def get_native_status(self):
        return self.runtime.get_epoch_status(), self.routing_status

    def dispatch(self, x, topk_idx, topk_weights, *, num_experts=384,
                 num_max_tokens_per_rank=None, expert_alignment=1,
                 do_cpu_sync=False, do_expand=True, previous_event=None,
                 async_with_compute_stream=False, allocate_on_comm_stream=False,
                 handle=None, defer_epilogue=False, **unsupported):
        if (unsupported or do_cpu_sync or not do_expand or expert_alignment != self.expert_alignment
                or async_with_compute_stream or allocate_on_comm_stream
                or defer_epilogue or handle is not None or num_experts != 384
                or num_max_tokens_per_rank not in (None, self.num_max_tokens_per_rank)):
            raise ValueError("SDAA dispatch requires expanded device-count routing with loaded alignment")
        if self._pending is not None:
            raise RuntimeError("combine the preceding EPHandle before reusing this buffer")
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
        ids = self._ids[:n]
        _sdaa.cast_indices(topk_idx, ids, self.routing_status, 384)
        layout = self._layout[:n + 1]
        send_counts = self._send_counts
        wire_x = self._wire_x[:n]
        wire_x[:, :5120].copy_(x)
        # Opaque metadata bytes: original row, not a numeric FP16 conversion.
        self.runtime.get_dispatch_layout(ids, topk_weights, wire_x, self.local_experts * 4,
                                         384, self.dp_size, self.dp_size, layout, send_counts)
        self.runtime.dispatch_data(ids, topk_weights, wire_x, self.dp_size, self.dp_size, send_counts)
        card_counts, card_ids = self._card_counts, self._card_ids
        card_weights, card_x = self._card_weights, self._card_x
        self.runtime.gather_dispatch_data(self.dp_rank, self.dp_size, card_counts, card_ids, card_weights, card_x)
        recv_x, recv_weights = self._expanded_x, self._expanded_weights
        rank_prefix, expert_prefix = self._rank_prefix, self._expert_prefix
        inverse, src = self._route_slots, self._source
        _sdaa.expand_dispatch(card_x, card_ids, card_weights, card_counts,
            recv_x, recv_weights, rank_prefix, expert_prefix,
            self._expert_counts, self._padded_counts, inverse, src,
            self.rank, self.routing_status, expert_alignment, self.dp_size)
        try:
            version = topk_idx._version
        except RuntimeError:
            version = None
        retained = (x, wire_x, ids, topk_weights, card_x, card_ids, card_weights, recv_x,
                    recv_weights, rank_prefix, expert_prefix, inverse, src,
                    card_counts, layout, send_counts, self.routing_status)
        handle = EPHandle(self, topk_idx, version, num_experts,
            self.num_max_tokens_per_rank, rank_prefix[-1], rank_prefix, expert_prefix,
            src, inverse, card_counts, layout, send_counts, tuple(x.shape), retained,
            self._expert_counts, self._padded_counts, expert_prefix[-1],
            torch.sdaa.current_stream(),
            expert_alignment=expert_alignment)
        self._pending = handle
        return recv_x, None, recv_weights, handle, EventOverlap(handle.stream, retained)

    def combine(self, x, handle, topk_weights=None, *, previous_event=None,
                async_with_compute_stream=False, allocate_on_comm_stream=False,
                defer_epilogue=False, **unsupported):
        handle.check(self)
        if self._pending is not handle:
            raise RuntimeError("EPHandle is no longer the active workspace owner")
        if torch.sdaa.current_stream() != handle.stream:
            raise ValueError("SDAA dispatch and combine must execute on the same stream")
        if (topk_weights is not None or unsupported or async_with_compute_stream
                or allocate_on_comm_stream or defer_epilogue
                or x.shape != (self.expanded_capacity, 5120) or x.dtype != torch.float16
                or x.device != self.device or not x.is_contiguous()):
            raise ValueError("SDAA combine requires preweighted FP16 expert outputs and the original handle")
        if previous_event is not None:
            previous_event.current_stream_wait()
        # The Ascend inverse-slot contract reduces local top-k slots once before
        # the native card reduction/IPC combine. Coefficients are already applied.
        card_output = self._card_output
        _sdaa.combine_expanded(x, handle.dst_buffer_slot_idx, card_output, self.routing_status)
        self.runtime.get_combine_layout(card_output, handle.card_counts, self.dp_size, self.dp_size)
        self.runtime.combine_data(card_output, self.dp_size, self.dp_size, handle.card_counts)
        output = self._output[:handle.original_shape[0]]
        self.runtime.gather_combine_data(self.dp_rank, self.dp_size, self.dp_size,
            handle.layout, handle.send_counts, output)
        retained = (handle, x, card_output, output)
        self._pending = None
        return output, None, EventOverlap(handle.stream, retained)
