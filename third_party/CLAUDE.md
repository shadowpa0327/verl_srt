## Purpose of directory under third_party
+ **ArcticInference_srt** - Suffix tree implementation for speculative decoding
+ **claude_docs** - Documentation for third-party integrations (vLLM, ArcticInference)

## Documentation Entry Point

**Start here**: [`claude_docs/INDEX.md`](claude_docs/INDEX.md) - Master navigation for all documentation

## Quick Links

| Task | Document |
|------|----------|
| Understand project goals | [`claude_docs/role_of_third_party_lib.md`](claude_docs/role_of_third_party_lib.md) |
| Understand vLLM architecture | [`claude_docs/skills/vLLM_v1_architecture.md`](claude_docs/skills/vLLM_v1_architecture.md) |
| Work with suffix tree cache | [`claude_docs/skills/arctic_inference_cache_managements.md`](claude_docs/skills/arctic_inference_cache_managements.md) |
| Measure acceptance length | [`claude_docs/skills/acceptance_length_measurement.md`](claude_docs/skills/acceptance_length_measurement.md) |
| Use prompt hash tree mapping | [`claude_docs/skills/prompt_hash_tree_mapping.md`](claude_docs/skills/prompt_hash_tree_mapping.md) |
| Load trees into vLLM | [`claude_docs/skills/vllm_suffix_tree_integration.md`](claude_docs/skills/vllm_suffix_tree_integration.md) |

## Source Documentation

| Component | Full Docs |
|-----------|-----------|
| vLLM V1 | [`claude_docs/VLLM_V1_ARCHITECTURE.md`](claude_docs/VLLM_V1_ARCHITECTURE.md) |
| ArcticInference | [`ArcticInference_srt/CLAUDE.md`](ArcticInference_srt/CLAUDE.md) |
| Parallel Cache API | [`ArcticInference_srt/claude_docs/PARALLEL_CACHE_GUIDE.md`](ArcticInference_srt/claude_docs/PARALLEL_CACHE_GUIDE.md) |
| Serialization Spec | [`ArcticInference_srt/claude_docs/SUFFIX_TREE_SERIALIZATION.md`](ArcticInference_srt/claude_docs/SUFFIX_TREE_SERIALIZATION.md) |

## Current Tasks
<!-- INSTRUCTION: Review and update this section when tasks are completed or new ones emerge -->
<!-- Last updated: 2025-12-10 -->

| Epic | Status | Plan |
|------|--------|------|
| _None currently_ | | |

### Completed

| Epic | Date | Plan |
|------|------|------|
| Hash Tree vLLM Integration | 2025-12-10 | [`task_plans/hash_tree_vllm_integration.md`](claude_docs/task_plans/hash_tree_vllm_integration.md) |
