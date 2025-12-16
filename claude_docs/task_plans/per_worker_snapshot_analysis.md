# Per-Worker Selective Snapshot Distribution Analysis

**Date**: 2025-12-16
**Status**: Analysis
**Related**: [`../skills/selective_snapshot_distribution.md`](../skills/selective_snapshot_distribution.md)

---

## Problem Statement

Current implementation of selective snapshot distribution reduces transfer size by only sending trees for the current batch. However, in multi-GPU/DP scenarios, **all workers receive the same snapshot** even though each worker only processes a subset of prompts.

### Example Scenario

```
Batch: 6 prompts [P1, P2, P3, P4, P5, P6]
DP Size: 2 workers

Current behavior:
  Controller → get_selective_snapshot([P1-P6]) → 6 trees
  Worker 1: receives all 6 trees, processes [P1, P2, P3]
  Worker 2: receives all 6 trees, processes [P4, P5, P6]

Ideal behavior:
  Worker 1: receives 3 trees for [P1, P2, P3]
  Worker 2: receives 3 trees for [P4, P5, P6]
```

### Memory Impact (Ray Object Store)

| Config | Current | Ideal | Waste Factor |
|--------|---------|-------|--------------|
| 2 workers | 2 × 6 trees | 6 trees total | 2x |
| 4 workers | 4 × 6 trees | 6 trees total | 4x |
| 8 workers | 8 × 6 trees | 6 trees total | 8x |

---

## Architectural Constraints

### Key Insight

**Workers do NOT have access to SuffixTreeManager** - it lives on the controller (ray_trainer.py).
Workers can only:
1. Receive data pushed from the controller (current)
2. Pull from Ray object store
3. Call external services (gRPC)

### Current Dispatch Modes

| Mode | Behavior |
|------|----------|
| `ONE_TO_ALL` | Same data sent to all workers |
| `DP_COMPUTE` | Different data per worker (caller pre-partitions into list of world_size) |
| `make_nd_compute_dataproto_dispatch_fn(mesh_name)` | Auto-chunks DataProto by DP rank |

---

## Option 1: Controller-Side Partitioning

### Overview

Controller replicates the DP partitioning logic and sends different snapshots to different workers.

### Implementation

```python
# In ray_trainer.py, before push_suffix_snapshot

# Step 1: Get DP partitioning info
mesh_name = "rollout"
if mesh_name not in self.actor_rollout_wg._dispatch_info:
    self.actor_rollout_wg._dispatch_info[mesh_name] = \
        self.actor_rollout_wg._query_dispatch_info(mesh_name)

dp_rank_mapping = self.actor_rollout_wg._dispatch_info[mesh_name]
dp_size = max(dp_rank_mapping) + 1
world_size = self.actor_rollout_wg.world_size

# Step 2: Partition hashes (same logic as DataProto.chunk via np.array_split)
batch_hashes = gen_batch.non_tensor_batch.get("prompt_hashes", np.array([]))
partitioned_hashes = np.array_split(batch_hashes, dp_size)

# Step 3: Create per-DP-rank snapshots
per_dp_data = []
for dp_rank in range(dp_size):
    worker_hashes = partitioned_hashes[dp_rank].tolist()
    s, m = self.suffix_tree_manager.get_selective_snapshot(worker_hashes)
    per_dp_data.append((s, m))

# Step 4: Map DP rank to worker
per_worker_snapshots = [per_dp_data[dp_rank_mapping[i]][0] for i in range(world_size)]
per_worker_mappings = [per_dp_data[dp_rank_mapping[i]][1] for i in range(world_size)]

# Step 5: Use DP_COMPUTE dispatch
self.actor_rollout_wg.load_suffix_snapshot_dp(per_worker_snapshots, per_worker_mappings)
```

### Pros
- Single-phase push
- Leverages existing `DP_COMPUTE` dispatch

### Cons
- **Tight coupling**: Must exactly replicate DP partitioning logic
- **Fragile**: Any change to `DataProto.chunk()` breaks this
- **Order-sensitive**: Hash order must match batch row order

---

## Option 2: Worker-Side Filtering

### Overview

Workers receive full data but only load trees for their prompts.

### Implementation

```python
# Worker side - in generate_sequences()
def generate_sequences(self, prompts: DataProto):
    if self._pending_snapshots:
        # Get hashes for our prompts (after DP chunking)
        prompt_hashes = prompts.non_tensor_batch.get("prompt_hashes", [])

        # Filter to only trees we need
        needed_trees = set()
        for h in prompt_hashes:
            if h in self._pending_hash_mapping:
                needed_trees.add(self._pending_hash_mapping[h])

        # Load only needed trees
        my_snapshots = [(idx, self._pending_snapshots[idx])
                        for idx in needed_trees if idx in self._pending_snapshots]

        self.rollout.load_suffix_snapshot(my_snapshots, filtered_mapping)
```

### Pros
- **Decoupled**: Workers determine their own needs
- **Correct by construction**: No partitioning replication

### Cons
- **Still transfers all data**: ONE_TO_ALL sends full snapshot
- **Worker state management**: Two-phase coordination

---

## Option 3: gRPC Snapshot Server (Recommended)

### Overview

Controller runs a gRPC server that serves tree snapshots on demand. Workers request only the trees they need after knowing their prompts.

### Architecture

```
Controller (ray_trainer.py)
    │
    ├─ SuffixTreeManager (owns forest data)
    │
    └─ gRPC Server: SnapshotService
        │
        │ GetTreeSnapshots(tree_indices) → snapshots
        │ GetHashMapping() → hash_to_tree_idx
        │
        v
[Worker 1]                      [Worker 2]
    │                               │
    │ After DataProto.chunk()       │ After DataProto.chunk()
    │ knows prompts [P1,P2,P3]      │ knows prompts [P4,P5,P6]
    │                               │
    │ Look up: [P1,P2,P3]→[0,1,2]   │ Look up: [P4,P5,P6]→[3,4,5]
    │ gRPC: GetTreeSnapshots([0,1,2])│ gRPC: GetTreeSnapshots([3,4,5])
    │                               │
    v                               v
  Load 3 trees                    Load 3 trees
```

### Existing Infrastructure

ArcticInference already has gRPC infrastructure in `arctic_inference/suffix_decoding/`:
- `server.py` - gRPC server (currently for speculation)
- `client.py` - gRPC client
- `proto/suffix_decoding.proto` - Protocol definitions

### New RPC Methods Needed

```protobuf
// Add to suffix_decoding.proto

service SuffixDecodingService {
  // ... existing RPCs ...

  // NEW: Get tree snapshots by indices
  rpc GetTreeSnapshots (GetTreeSnapshotsRequest) returns (GetTreeSnapshotsResponse) {}

  // NEW: Get hash mapping for tree lookup
  rpc GetHashMapping (Empty) returns (HashMappingResponse) {}
}

message GetTreeSnapshotsRequest {
  repeated int32 tree_indices = 1;
}

message TreeSnapshot {
  int32 tree_idx = 1;
  bytes data = 2;
}

message GetTreeSnapshotsResponse {
  repeated TreeSnapshot snapshots = 1;
}

message HashMappingResponse {
  map<string, int32> hash_to_tree_idx = 1;
}
```

### Server Implementation

```python
# Add to server.py

class SuffixDecodingServicer(suffix_decoding_pb2_grpc.SuffixDecodingServiceServicer):
    # ... existing methods ...

    def GetTreeSnapshots(self, request, context):
        """Return snapshots for requested tree indices."""
        try:
            tree_indices = list(request.tree_indices)
            snapshots = self.cache.create_selective_snapshot(tree_indices)

            response = suffix_decoding_pb2.GetTreeSnapshotsResponse()
            for tree_idx, data in snapshots:
                snapshot = response.snapshots.add()
                snapshot.tree_idx = tree_idx
                snapshot.data = bytes(data)
            return response
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return suffix_decoding_pb2.GetTreeSnapshotsResponse()

    def GetHashMapping(self, request, context):
        """Return hash-to-tree-index mapping."""
        try:
            mapping = self.cache._hash_to_tree_idx
            return suffix_decoding_pb2.HashMappingResponse(hash_to_tree_idx=mapping)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return suffix_decoding_pb2.HashMappingResponse()
```

### Client Implementation

```python
# Add to client.py

class SuffixDecodingClient:
    # ... existing methods ...

    def get_tree_snapshots(self, tree_indices: List[int]) -> List[Tuple[int, bytes]]:
        """Fetch snapshots for specific tree indices."""
        request = suffix_decoding_pb2.GetTreeSnapshotsRequest(tree_indices=tree_indices)
        response = self.stub.GetTreeSnapshots(request)
        return [(s.tree_idx, s.data) for s in response.snapshots]

    def get_hash_mapping(self) -> Dict[str, int]:
        """Fetch hash-to-tree-index mapping."""
        response = self.stub.GetHashMapping(suffix_decoding_pb2.Empty())
        return dict(response.hash_to_tree_idx)
```

### Integration Flow

```python
# Controller (ray_trainer.py)
class RayPPOTrainer:
    def __init__(self, ...):
        # Start gRPC server if suffix tree enabled
        if self.suffix_tree_config.enable:
            self._start_snapshot_server()

    def _start_snapshot_server(self):
        """Start gRPC server for snapshot distribution."""
        import grpc
        from concurrent import futures

        self._grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        # Use existing servicer with our cache
        servicer = SuffixDecodingServicer(cache=self.suffix_tree_manager._cache)
        suffix_decoding_pb2_grpc.add_SuffixDecodingServiceServicer_to_server(
            servicer, self._grpc_server
        )
        self._grpc_port = self._grpc_server.add_insecure_port('[::]:0')  # Auto-assign port
        self._grpc_server.start()

        # Push server address to workers
        self.actor_rollout_wg.set_snapshot_server(f"localhost:{self._grpc_port}")

# Worker (fsdp_workers.py)
class ActorRolloutRefWorker:
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_snapshot_server(self, server_address: str):
        """Store gRPC server address for snapshot fetching."""
        self._snapshot_server_address = server_address
        self._snapshot_client = SuffixDecodingClient.from_address(server_address)

    def generate_sequences(self, prompts: DataProto):
        # After DataProto is chunked, we know our prompts
        if hasattr(self, '_snapshot_client'):
            prompt_hashes = prompts.non_tensor_batch.get("prompt_hashes", [])

            # Get hash mapping (cached or fresh)
            if not hasattr(self, '_hash_mapping'):
                self._hash_mapping = self._snapshot_client.get_hash_mapping()

            # Determine needed tree indices
            needed_indices = set()
            for h in prompt_hashes:
                if h in self._hash_mapping:
                    needed_indices.add(self._hash_mapping[h])

            # Fetch only needed trees via gRPC
            if needed_indices:
                snapshots = self._snapshot_client.get_tree_snapshots(list(needed_indices))
                filtered_mapping = {h: idx for h, idx in self._hash_mapping.items()
                                   if idx in needed_indices}
                self.rollout.load_suffix_snapshot(snapshots, filtered_mapping)

        # Continue with generation...
```

### Advantages

| Aspect | Benefit |
|--------|---------|
| **True pull model** | Workers request exactly what they need |
| **Decoupled** | No DP partitioning replication |
| **Scalable** | gRPC handles concurrent requests efficiently |
| **Existing infra** | Leverages ArcticInference gRPC setup |
| **Extensible** | Easy to add compression, streaming, caching |
| **Debuggable** | Standard gRPC tooling (grpcurl, etc.) |

### Considerations

| Aspect | Notes |
|--------|-------|
| **Latency** | ~1-5ms per gRPC call (local), acceptable for batch generation |
| **Concurrency** | Workers fetch in parallel, server handles with thread pool |
| **Caching** | Hash mapping can be cached on workers |
| **Failure handling** | Standard gRPC retry/timeout mechanisms |
| **Network** | Local loopback is fast; cross-node needs consideration |

### Performance Optimization

1. **Cache hash mapping**: Workers cache mapping, only refresh periodically
2. **Batch tree requests**: Single gRPC call for all needed trees
3. **Async fetch**: Overlap with other initialization
4. **Streaming**: For very large snapshots, use gRPC streaming

---

## Comparison Summary

| Aspect | Option 1 (Controller) | Option 2 (Worker Filter) | Option 3 (gRPC) |
|--------|----------------------|--------------------------|-----------------|
| Transfer efficiency | Optimal | Same as before | Optimal |
| Implementation complexity | High | Medium | Medium |
| Correctness risk | High (coupling) | Low | Low |
| Existing infrastructure | None | None | **gRPC exists** |
| Maintainability | Fragile | Good | Good |
| Debugging | Hard | Medium | Easy (gRPC tools) |
| Extensibility | Limited | Limited | **High** |

---

## Recommendation

**Option 3 (gRPC Snapshot Server)** is recommended because:

1. **Leverages existing infrastructure**: ArcticInference already has gRPC server/client
2. **True pull model**: Workers request exactly what they need - no coupling to DP logic
3. **Clean separation**: Controller owns data, workers own requests
4. **Extensible**: Easy to add compression, caching, metrics
5. **Standard tooling**: gRPC has excellent debugging and monitoring tools

### Implementation Plan

1. **Phase 1**: Add new proto messages and RPCs (GetTreeSnapshots, GetHashMapping)
2. **Phase 2**: Implement server methods in `server.py`
3. **Phase 3**: Implement client methods in `client.py`
4. **Phase 4**: Integrate server into SuffixTreeManager/ray_trainer
5. **Phase 5**: Update workers to use gRPC client for snapshot fetching
6. **Phase 6**: Add metrics and monitoring

### Estimated Effort

| Component | Effort |
|-----------|--------|
| Proto changes | Small (~20 lines) |
| Server implementation | Medium (~50 lines) |
| Client implementation | Small (~30 lines) |
| Controller integration | Medium (~50 lines) |
| Worker integration | Medium (~40 lines) |
| Tests | Medium (~100 lines) |
| **Total** | ~290 lines |
