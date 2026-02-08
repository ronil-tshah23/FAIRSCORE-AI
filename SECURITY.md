# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible for receiving such patches depend on the CVSS v3.0 Rating:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of FairScore seriously. If you believe you have found a security vulnerability, please report it to us responsibly.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to the maintainers:

- **Email**: security@fairscore.example.com
- **Subject**: [SECURITY] Brief description of the vulnerability

### What to Include

Please include the following information in your report:

1. **Type of vulnerability** (e.g., buffer overflow, SQL injection, XSS, etc.)
2. **Full paths of affected source files**
3. **Step-by-step instructions to reproduce** the vulnerability
4. **Proof-of-concept or exploit code** (if possible)
5. **Impact assessment** – What an attacker could accomplish if exploited
6. **Suggested fix** (if you have one)

### Response Timeline

- **Initial Response**: Within 48 hours of receiving your report
- **Vulnerability Confirmation**: Within 7 days
- **Patch Development**: Depends on severity
  - Critical: Within 24-48 hours
  - High: Within 7 days
  - Medium: Within 14 days
  - Low: Within 30 days
- **Public Disclosure**: Coordinated with reporter after patch release

### What to Expect

1. **Acknowledgment**: We will acknowledge receipt of your vulnerability report
2. **Communication**: We will keep you informed of our progress
3. **Credit**: If you want, we will give you credit in the release notes
4. **No Retaliation**: We will not take legal action against you if you follow responsible disclosure practices

## Security Best Practices for Users

### API Keys

- **Never commit API keys** to version control
- Use environment variables for sensitive credentials:
  ```bash
  set GOOGLE_API_KEY=your_api_key
  ```
- Consider using a secrets manager for production deployments

### Data Handling

- FairScore processes sensitive financial data
- Ensure proper data encryption at rest and in transit
- Follow your organization's data retention policies
- Do not share model outputs with unauthorized parties

### Model Security

- Pre-trained models are included in the distribution
- Verify model checksums if provided
- Do not load models from untrusted sources
- Regularly update to the latest version for security patches

## Known Security Considerations

### PDF Processing

- PDF parsing uses Google Gemini AI
- Ensure you trust the PDFs being processed
- PDF content is sent to Google's API – review their privacy policy

### Local Model Storage

- Models are stored as `.pkl` (pickle) files
- Only load pickle files from trusted sources
- Pickle deserialization can execute arbitrary code

## Security Updates

Security updates will be announced through:

- GitHub Security Advisories
- Release notes in the repository
- Email to registered security contacts (if applicable)

## Contact

For any security-related questions that don't involve vulnerabilities, please open a GitHub issue with the `security` label.

---

Thank you for helping keep FairScore and our users safe! 🔒
