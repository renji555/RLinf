# Server development backup

This snapshot preserves Cosmos3 SFT, checkpoint conversion, DAgger, related
configuration, and tests from /mnt/public2/rhy/cosmos_rlinf/RLinf.

The Git branch contains the RLinf working files. manifest.json records the
original base commit and SHA-256 hashes of the 18 development files.

## Restore the framework change

Install the dependencies required by this RLinf checkout. Restore the framework
at the commit recorded in manifest.json, then apply its local change:

    git -C .venv/cosmos-framework apply ../../server-backup/cosmos-framework.patch

The patch points the Nano tokenizer at
/mnt/public2/rhy/docker_dev/update/model/Qwen3-VL-8B-Instruct.
Set that path for the new machine before running a job.

Model weights, training data, logs, checkpoints, and the virtual environment are
not included. Paths in the training and evaluation YAML files must also be
updated for the new machine. The separate fuxu_rlinf and test checkouts are
outside this backup.
