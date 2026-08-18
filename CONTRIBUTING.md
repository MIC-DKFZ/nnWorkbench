# Contributing

Keep `nnWorkbench` focused on transparent nnU-Net workflows and nnInteractive-powered annotation. Integrations with viewers, nnInteractive, and nnU-Net should be isolated behind adapters, documented with compatible tool versions, and tested without requiring patient data.

Before submitting changes, run `pytest` and `ruff check .`. Do not commit imaging data, scribbles, labels, credentials, or site-specific configuration.
