# Security notes

This is a locally run research app. The provided configuration binds to loopback (`127.0.0.1`); it is not a hardened public upload service. Model loading uses local Safetensors/checkpoint files, and pinned download scripts verify recorded hashes.

Never post tokens, private imagery, precise site coordinates or personal server logs in a public issue. If GitHub's private vulnerability reporting is enabled, use the repository's **Security → Report a vulnerability** route. Otherwise contact the maintainer through an available private contact on their GitHub profile before sharing sensitive details. Ordinary nonsensitive bugs belong in Issues.

The supported release line is `0.2.x`. Third-party dependency or model vulnerabilities should also be reported through the relevant upstream project's process.
