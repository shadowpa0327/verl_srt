## Context

Currently, at `/home/ubuntu/verl_srt/recipe/srt/shared_memory_cache_manager.py` we have the implementations for SharedMemoryCache manager. 

This manager take rollout results from two sources. From primary rollout using `update_from_rollout` and from secondary rollout using `update_from_secondary`. 

From may carefully review, I have concerns on the inconsistent interface. Particularily, in the `update_from_secondary`, we send the update to the cache servers by forcing the response_per_prompt=1. I am not sure whether this will sliently results in missing some of the updates to the Tree and thereby leading to sub-optimial speculation quality....

## Goal and tasks

- Create a stand along scripts. Follow the similar 3-tick testing style with `/home/ubuntu/verl_srt/tests/workers/rollout/rollout_vllm/test_runahead_suffix_effectiveness.py`, This scrtips should
    - Follow how we prepare data for `recipe.srt.main_ppo` using DAPO datasets (ref: `/home/ubuntu/verl_srt/recipe/srt/scripts/dapo/run_dapo_srt_runahead_shm.sh`)
    - Perform rollout and do cache updates with the secondary results. 
    - We wanna to explore whether our training loops right now have some silent bug that hinder the trigger for speculative decoding.