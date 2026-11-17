---
id: incident-response-runbook
title: Security Incident Response Runbook
version: 5
status: approved
owner: sam
approved_by: priya
org: it
audience: it_engineer, security_lead
effective: 2026-05-01
---
# Security Incident Response Runbook

## Severity 1 incidents
For a Sev1 security incident, page the on-call security engineer, who must respond within 15 minutes. Open a dedicated incident channel and assign an incident commander.

## Evidence
Preserve logs and disk images before restarting or reimaging any affected system.

## Patient data
If patient health information may have been exposed, notify the Compliance Officer within 24 hours so breach notification rules can be assessed.

## Break-glass access
Emergency admin credentials are stored in the sealed vault. Using them requires a second engineer to approve the request in the access tool.
