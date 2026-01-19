In our SRT developments, so far we default the sliding window-based metrics for data feeding. 
For intance,
+ step 1: we have the batch of step 2 become secondary
+ step 2: we have the batch of step 3 become third. 

People have commonly use DAPO, GRPO for testing. The core impact of DAPO and GRPO on rollout is that it will repeat the samples/prompts for `n` times. 

I think one design angle we should explore is the combination ratio of the unique sample numbers and its repeatness `n`. 

To elaborate, if our run-ahead can totally process extra 256 secondary samples every rollout, shall we pick 16 samples and each with n=16? Or, shall we prioritize the exploration first to run as many unique samples as possible, even the one in more far away batches (e.g., step 1 --> run step 4's batch). 