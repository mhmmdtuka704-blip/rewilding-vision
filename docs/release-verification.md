# Public-release verification

Publication preparation: **28 October 2026**. The scientific benchmark remains the recorded 25 October run.

## Checks completed locally

- **45 unit/regression tests passed** in the publishing copy, including all 40 original cases plus five publication regressions.
- A dataset-free Streamlit smoke run passed with real synthetic uploaded bytes, ExG analysis, six metrics, binary tabs, result persistence, corrupt-input handling and GeoTIFF export reopening.
- The additional tests verify checksum rejection without overwriting existing files, valid-cache reuse without network, atomic verified download, startup without dataset/corporate logo files, and split/manifest consistency.

The smoke fixtures are synthetic software controls, not real TOD scenes, genuine geographic site coordinates or extra accuracy measurements. CI runs these same offline software checks; the GitHub workflow badge reports the actual remote status, not a fabricated successful build.

The pinned checkpoint setup verified all recorded model-file hashes. The **actual neural `--ml` smoke run passed** on the same synthetic software fixtures, including ordinary-image analysis and GeoTIFF export. This exercises inference and integration, not additional accuracy.

The repository checker passed for local documentation links, file-size/path/credential-pattern scanning, reconstruction of recorded metrics from confusion counts, and the exact dataset manifest hash. `.gitattributes` preserves the manifest's bytes so Windows/Linux checkout cannot alter the recorded checksum.

See [results](results.md) for measured model performance and [reproducibility](reproducibility.md) for the original experiment.
