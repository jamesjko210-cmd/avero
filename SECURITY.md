# Security Policy

Avero V4 is a macOS, single-owner functional preview. It is not designed as a hosted service,
shared-account system, or security boundary between mutually untrusted users on the same Mac.

## Report a vulnerability privately

Use GitHub's private vulnerability reporting for this repository:

1. Open the repository's **Security** tab.
2. Select **Advisories** and then **Report a vulnerability**.
3. Describe the affected public revision, impact, and the smallest safe reproduction.

Do not open a public issue containing vulnerability details, credentials, private data, local file
contents, or third-party account identifiers. If private vulnerability reporting is unavailable,
do not publish the sensitive report; wait until a private reporting channel is available.

Use synthetic values in reproductions. Remove tokens, cookies, account identifiers, message
contents, local paths, and screenshots of private applications. A useful report explains the trust
boundary that was crossed and whether an external side effect occurred, but it should not trigger
another side effect merely to collect evidence.

## Preview security boundary

- Run Avero only under the intended local macOS account and keep runtime configuration outside the
  repository with owner-only permissions.
- Treat approvals as single-use authorization for one exact action. An unknown outcome is not
  success and must not be replayed automatically.
- Connector, browser, accessibility, microphone, calendar, and automation permissions remain under
  the operating system or third-party service. Review those prompts yourself.
- Do not use the preview as a sandbox for hostile code or as a multi-user secrets service.

Security reports are reviewed on a best-effort basis. This preview currently has no response-time,
support-lifetime, or bug-bounty commitment.
