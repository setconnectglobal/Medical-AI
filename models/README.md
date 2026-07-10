# Model Assets

Production weights are tracked in the repo at:

- `specialist_models/` — 7 category-specific specialist CNNs
- `rl_agent_weights (1).pth` — generalist ResNet-50 weights
- `rl_training_metadata (1).json` — RL Q-table

You can also place optional copies here for local overrides. Set `MODEL_DIR=./models` only if you use this folder exclusively.

On startup, `app.py` prints ✓/✗ for each weight file so you can verify everything loaded.
