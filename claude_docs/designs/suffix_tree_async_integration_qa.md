# Suffix Tree Integration for Async Rollout - Design Questions

Please fill in your answers below. These decisions will guide the implementation.

---

## Q1: Snapshot Distribution Timing

**Context:** In sync mode, snapshots are pushed immediately before `generate_sequences()`. In async mode, we have more flexibility.

**When should snapshots be pushed to workers in async mode?**

- [ ] A. Before `wake_up()` - Push snapshot, then wake up workers
- [ ] B. During `wake_up()` - Combine snapshot loading with wake up
- [ ] C. Separate call before generate - Explicit `load_suffix_snapshot()` call (same as sync mode)
- [ ] D. Other: _______________

**Your answer:**

---

## Q2: Rollout Mode Support

**Context:** Async rollout supports three modes: HYBRID (workers in same process), COLOCATED (same placement group), STANDALONE (separate resources).

**Which modes should we support for suffix tree integration?**

- [ ] A. HYBRID only (most common, simplest implementation)
- [ ] B. HYBRID + COLOCATED
- [ ] C. All modes (HYBRID + COLOCATED + STANDALONE)
- [ ] D. Other: _______________

**Your answer:**

---

## Q3: Communication Mechanism for Non-HYBRID Modes

**Context:** In HYBRID mode, we can call `worker.load_suffix_snapshot.remote()` directly. In other modes, workers communicate via ZMQ.

**For COLOCATED/STANDALONE modes, how should we push snapshots?**

- [ ] A. Skip for now - Only implement HYBRID mode
- [ ] B. Add new ZMQ method type for snapshot loading
- [ ] C. Store snapshot in server, apply during next worker initialization
- [ ] D. Push via HTTP endpoint on vLLMHttpServer
- [ ] E. Other: _______________

**Your answer:**

---

## Q4: Hash Computation Location

**Context:** Hash needs to be computed for each prompt to enable tree lookup during speculative decoding.

**Where should prompt hash be computed?**

- [ ] A. In `vLLMHttpServer.generate()` - Before calling engine
- [ ] B. In `SingleTurnAgentLoop.run()` - Before calling server
- [ ] C. In the vLLM engine (existing patches)
- [ ] D. Other: _______________

**Your answer:**

---

## Q5: Multi-Turn Conversation Handling

**Context:** In async mode with agent loops, prompts can grow across turns (user → assistant → tool → assistant).

**How should we handle hash computation for multi-turn conversations?**

- [ ] A. Compute hash on the full accumulated prompt before each LLM generation
- [ ] B. Compute hash only on the initial user prompt (ignore tool responses)
- [ ] C. Compute hash on the last N tokens (sliding window)
- [ ] D. Don't support suffix trees for multi-turn (single-turn only)
- [ ] E. Other: _______________

**Your answer:**

---

## Q6: Tool Response Handling

**Context:** In multi-turn conversations, tool responses are injected between LLM generations.

**Should tool response tokens be included in suffix tree updates?**

- [ ] A. Yes - Include all tokens (LLM + tool responses)
- [ ] B. No - Only include LLM-generated tokens (use response_mask)
- [ ] C. Configurable via config option
- [ ] D. Other: _______________

**Your answer:**

---

## Q7: Snapshot Scope

**Context:** We can push full snapshots (all trees) or selective snapshots (only trees for current batch).

**What snapshot strategy should we use for async mode?**

- [ ] A. Full snapshot - Push all trees to all replicas
- [ ] B. Selective snapshot - Only trees matching current batch hashes (same as sync mode)
- [ ] C. Per-replica selective - Each replica gets only trees for prompts it will process
- [ ] D. Other: _______________

**Your answer:**

---

## Q8: Sticky Session Interaction

**Context:** `AsyncLLMServerManager` uses sticky sessions to send multi-turn chats to the same server for prefix caching.

**How should suffix trees interact with sticky sessions?**

- [ ] A. No special handling - Hash lookup works independently
- [ ] B. Push snapshot only to the sticky session server
- [ ] C. Disable suffix trees when sticky sessions are active
- [ ] D. Other: _______________

**Your answer:**

---

## Q9: Agent Loop Types

**Context:** There are multiple agent loop types: `SingleTurnAgentLoop`, `ToolAgentLoop`, etc.

**Which agent loops should support suffix trees?**

- [ ] A. SingleTurnAgentLoop only (simplest case)
- [ ] B. All agent loops
- [ ] C. Configurable per agent loop type
- [ ] D. Other: _______________

**Your answer:**

---

## Q10: Memory/Performance Tradeoffs

**Context:** Suffix tree snapshots can be large. Broadcasting to all replicas creates copies.

**What optimizations should we consider?**

- [ ] A. No optimization needed - Snapshots are small enough
- [ ] B. Add compression for snapshots
- [ ] C. Implement incremental updates instead of full snapshots
- [ ] D. Lazy loading - Only load trees on first use
- [ ] E. Other: _______________

**Your answer:**

---

## Q11: Error Handling

**Context:** Snapshot loading might fail on some workers but succeed on others.

**How should we handle partial failures?**

- [ ] A. Fail fast - If any worker fails, abort generation
- [ ] B. Best effort - Continue with workers that succeeded
- [ ] C. Retry with backoff - Attempt to reload on failed workers
- [ ] D. Other: _______________

**Your answer:**

---

## Q12: Configuration

**Context:** We need to decide how to configure suffix trees for async mode.

**Should we use the same config as sync mode or add async-specific options?**

- [ ] A. Same config - `suffix_decoding.enable: true` enables for both modes
- [ ] B. Add async-specific flag - `suffix_decoding.enable_async: true`
- [ ] C. Mode-aware config - Different defaults for sync vs async
- [ ] D. Other: _______________

**Your answer:**

---

## Q13: Implementation Priority

**Context:** We have limited resources and need to prioritize.

**What's the implementation priority?**

- [ ] A. Full feature parity with sync mode
- [ ] B. Minimal viable implementation (SingleTurn + HYBRID only)
- [ ] C. Focus on performance optimization
- [ ] D. Other: _______________

**Your answer:**

---

## Q14: Additional Requirements

**Any additional requirements or constraints not covered above?**

**Your answer:**

---

## Q15: Testing Requirements

**What level of testing is required before deployment?**

- [ ] A. Unit tests only
- [ ] B. Unit tests + integration tests
- [ ] C. Full e2e test with actual training run
- [ ] D. Performance benchmarks required
- [ ] E. Other: _______________

**Your answer:**

---

## Summary of Your Choices

After filling in, please summarize your key decisions here:

1. **Timing:**
2. **Modes supported:**
3. **Communication:**
4. **Hash location:**
5. **Multi-turn:**
6. **Priority:**

---

*Please return this document with your answers filled in.*
