# Publication choices

The repository is arranged around what a reviewer can verify: the practical problem, a readable entry point, a repeatable setup, inspectable modules, automated tests, evidence and honest limits.

The preparation followed GitHub's own guidance on [README content](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes), [Mermaid diagrams](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-diagrams), and [Python CI](https://docs.github.com/en/actions/tutorials/build-and-test-code/python). Those sources support documentation/automation patterns; they are not evidence of a particular employer's hiring criteria.

- The root README is the introduction; technical detail lives in linked documents.
- Diagrams explain data flow, failure paths and transfer learning, rather than decorating unsupported claims.
- The metrics table links to raw counts and describes the exact split.
- CI checks behavior on software fixtures without downloading or redistributing the dataset.
- Model/data cards separate measured facts from unvalidated use cases.
- Public files omit local paths, uploaded imagery, server logs, caches, old presentation bundles and large raw data.
- Third-party model restrictions and dataset attribution are retained. The repository does not pretend that a permissive code label grants commercial model rights.

The project demonstrates applied ML integration, resource-aware training, spatial reasoning, defensive error handling, GIS interoperability and reproducibility. It remains a research application, not an externally validated ecological monitoring product.
