# Context

We have collected the primary rollout and secondary rollouts. We used the secondary to filled the trees and perform speculative decoding for the next steps.
Now, we have stored some rollout datas. Explore `/home/ubuntu/verl_srt/rollout_datas/DAPO/DAPO-Qwen2.5-7b-MATH-SRT-Runahead` for examples. 

I want to build a simulator, that read from collected rollout datas. Use the SRT data flow logics to fill the cache and perform speculation and simulate real decoding.
Particularily, in the rollout data we got the completed response. What I want you to do is for every sequects start from its promts, perform speculation got the speculated tokens, check with the response to verify the matchness untill the total responses generated. 

