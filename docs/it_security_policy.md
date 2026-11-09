# IT Security and Access Policy

## Accounts and passwords
Every employee gets a company Google Workspace account on their first day. Passwords must be at least 14 characters and stored in the company password manager (1Password). Two-factor authentication with a hardware key or authenticator app is mandatory for Google Workspace, GitHub, AWS and the CRM; SMS codes are not accepted.

Shared accounts are not allowed. If a tool only supports one login, the owner stores the credential in a shared 1Password vault and access is reviewed quarterly.

## Devices
Company laptops are managed by Kandji. Full-disk encryption, automatic screen lock after 5 minutes and automatic updates are enforced. Personal devices may read company email only through the Google app with a work profile; they may not access the production VPN.

Lost or stolen devices must be reported to it-help@company.example and on the #it-security Slack channel within 24 hours so the device can be wiped remotely.

## Access requests
Request access to systems through the IT Help Desk form in Slack. Access to production databases and AWS production accounts needs the approval of the engineering manager and is granted for 90 days at a time. All access is removed on the last working day; managers confirm the offboarding checklist with IT.

## Data handling
Customer data is classified as Confidential. It may only be stored in approved systems (the production database, the data warehouse, and the CRM) and never in personal drives, spreadsheets emailed around, or AI tools that have not been approved by security. Exports for analysis must be anonymised.

## Phishing and incidents
Report suspicious emails with the "Report phishing" button in Gmail. If you clicked a link or entered a password, change the password immediately and tell security@company.example; nobody is penalised for reporting quickly. Security incidents are triaged within one hour during business hours and within four hours otherwise.

## Vendor and software approval
New SaaS tools that touch customer data need a security review before purchase. Submit the vendor review form; reviews take about five working days. Free tools used only with public data can be adopted without review but must be listed in the tools register.
