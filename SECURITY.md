# Security Policy

## Reporting a vulnerability

Please report security issues **privately**, not via public issues/PRs.

- Use GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
  ("Report a vulnerability" on the Security tab), or
- email the maintainers listed in `CODEOWNERS`.

We aim to acknowledge within a few business days and will coordinate a fix and disclosure
timeline with you.

## Scope

This repository is research/forecasting code and curated public data. It contains **no
credentials** by policy. If you find a committed secret, local absolute path, or other
sensitive artifact, treat it as a vulnerability and report it privately so we can purge it.

## For maintainers — pre-publish hygiene

- CI runs a secret scan on every PR; do not merge on a finding.
- Cloud access is environment-configured via `MBARI_*` variables — never hardcode a project
  id, key, or token.
- Generated reports and raw data caches are gitignored; never force-add them.
