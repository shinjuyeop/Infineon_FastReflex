# Repository Development Rules

이 문서의 규칙은 repository 전체에 적용되며 naming convention의 authoritative source다.

## Canonical files and naming

- Before creating any new file, first decide whether the information or implementation belongs in an existing canonical file.
- Use lowercase `snake_case` for Python modules, packages, directories, and role-based config names. Do not put dates in Python source filenames.
- Do not use filename suffixes such as `v1`, `v2`, `v3`, `final`, `final2`, `latest`, `new`, `revised`, `backup`, or `old` for source, config, or canonical documentation.
- One responsibility should have one canonical implementation. Reuse or modify it before splitting files; split only when responsibilities are genuinely different and do not introduce premature abstractions.
- Git history is the archive. Delete unused code instead of keeping versioned, `old`, or `backup` copies.
- Keep directories lowercase `snake_case`. Do not create speculative `old`, `backup`, `archive`, `final`, versioned, or experimental directories.

The canonical current-state documents are:

- `docs/architecture.md`
- `docs/dataset.md`
- `docs/experiment_protocol.md`

Update these files in place. Do not create dated or version-suffixed replacements.

- Create a decision/checkpoint history document only when it adds lasting value, using `docs/milestones/YYYYMMDD_title.md` with an Asia/Seoul date and lowercase `snake_case` title. Do not create one for every milestone or duplicate canonical docs at length.
- Name actual experiment/analysis reports `reports/YYYYMMDD_title.md`. Keep design contracts and current architecture in canonical docs, not reports.

## Configuration and experiments

- Keep one role-based canonical config per dataset type, such as `configs/dataset/hazard.yaml` or `configs/dataset/terrain.yaml`. Store schema versions inside the config and provenance rather than its filename.
- Keep the dataset-type config separate from generated dataset identity. Record `dataset_id`, `created_at`, `source_commit`, and `schema_version` in manifest metadata, and do not use meaningless `final` or `latest` identifiers.
- Add model-family configs only when needed, using canonical names such as `mlp.yaml`, `cnn1d.yaml`, `gru.yaml`, or `lstm.yaml`. Hyperparameter changes do not justify filename variants.
- Record reproducible experiments as `configs/experiment/YYYYMMDD_title.yaml` using an Asia/Seoul date and lowercase `snake_case` title. An experiment config should reference canonical dataset/model configs and record window, seed, split, and training settings.
- Do not create a new Python runner for each experiment. Reuse the canonical CLI and source modules, and express experiment differences in YAML.
- Do not create files, directories, frameworks, packages, or abstractions before they are needed.

## Repository and artifact boundaries

- Do not commit generated datasets or arbitrary experiment outputs. Keep generated data, models, and reports outside source directories.
- Keep dataset, model, output, and configuration provenance explicit and separate.
- Use gitignored locations for intermediate artifacts. Preserve only explicitly reviewed frozen Research-to-Deployment releases under `artifacts/releases/`; define release naming when the first real release exists.
- Do not put research training logic in the E84 repository.
- Do not put quantization, firmware, or HIL logic in this research repository.
- Keep README `Current Status` synchronized with the actual project state; it is the canonical status.
- Keep the complete pipeline understandable within five minutes to a developer learning the codebase.
- Prefer explicit, simple code over clever abstractions.
- Do not silently copy code, datasets, models, outputs, or dependencies from the legacy `/d/shin/Infineon` repository. Any future migration must be explicit and reviewed.
- Preserve artifact provenance when a frozen model is eventually exported to the deployment repository.
