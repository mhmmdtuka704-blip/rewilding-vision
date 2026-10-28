# Contributing

Start with a concrete problem or reproducible failure. Keep changes focused: the aim is dependable comparison, not another unmeasured model or a GIS platform.

1. Install the development environment from the README.
2. Create a branch and describe the behavior being changed.
3. Run `python -m pytest -q`, `python scripts/verify_app.py`, and `python scripts/check_repository.py`.
4. Add regression coverage when the change affects correctness, validity masks, geometry, model loading or exports.
5. Include meaningful verification and limitations in the pull request.

Do not commit local uploads, precise site coordinates, credentials, raw RUGD files, base checkpoints, virtual environments or generated server logs. A synthetic fixture is appropriate for a software regression; it is not an accuracy benchmark.

Model changes need an explicit training recipe and independent evaluation. The published test split has already been observed; do not tune on it and present the result as a fresh held-out measurement. Contributions remain subject to the repository and upstream rights described in LICENSE and THIRD_PARTY_NOTICES.md.
