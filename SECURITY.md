# Security Policy

## Supported Versions
Only the latest version of the Monterey Bay AI Lab is supported for security updates.

## Reporting a Vulnerability
We take the security of our coastal intelligence platform seriously. If you find a vulnerability, please do NOT open a public issue. Instead:
1.  Contact the maintainers directly.
2.  Provide a detailed report and reproduction steps.
3.  Allow us time to remediate before public disclosure.

## Focus Areas
- **Secret Prevention:** We rigorously scan for BigQuery keys and GCP service account JSONs.
- **Dark Machine Protection:** Our local Forge is private-by-design; inbound vulnerabilities are our highest priority.
- **Container Hygiene:** Docker builds must not include local credentials, raw private datasets, generated caches, or workstation-only artifacts.
- **Open-Source Boundary:** Public issues and pull requests should avoid sensitive operational details, non-public data paths, and private cloud identifiers.

---
**Monterey Bay AI Lab — Protecting Coastal Intelligence.**
