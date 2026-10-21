# training/mdn: mixture density network baseline

Training code for the mixture-density planning baseline (not written yet).

The policy has to fit the same slot as the other learned planners: input is what `planners/_torch_policy.py` builds from the
belief (`meanvarmarker_map` [B,3,51,51], `current_position` [B,3], `total_variance_condition` [B,1]); output is a mixture over
the 8 control waypoints (normalized xyz in [-1, 1], shape [3, 8]) instead of one regressed set (`training/imitatetrans`) or a
denoised sample (`training/diffusion`).

Data: the same expert dataset the other two use (HIGHOPT_DATASET_NAIP_141026.pt), whose multimodality (several CMA-ES runs per
decision point) is what a mixture model is meant to capture.

Open design points: number of mixture components K, covariance structure per component (diagonal over the 24 values, or per
waypoint), the loss (negative log-likelihood), how a single plan is picked at run time (sample a component, or take the most
likely one), and whether to reuse the imitation network's backbone.
