# V3 deployment test snapshot

The `params/` files contain the interface/model settings used by the
`2026-08-05_04-55-55_realcam_crop_noise_l2far_render12hz_from42000`
`model_68500.pt` regression checks. `env.yaml` retains only the deployment
interface fields, policy timing, depth-camera mode and robot pose/limit fields;
`agent.yaml` retains the saved runner configuration.

No checkpoint weights, images or training logs are included. The test harness
uses `model_68500.pt` as a path placeholder and mocks runner loading; that file
does not need to exist. These fixtures are test data, not a substitute for the
full `params/` distributed with a real deployment checkpoint.

When the original local training snapshots are available, the tests use them;
a clean checkout falls back to these fixtures. The suite does not open hardware
or start Isaac Sim.
