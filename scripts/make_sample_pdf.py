"""Generate docs/remote_work_policy.pdf once (reportlab, dev dependency) so the PDF reading path is exercised."""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parents[1] / "docs" / "remote_work_policy.pdf"
SECTIONS = [
    ("Remote Work Policy", None),
    ("Eligibility", "All employees may work remotely up to three days per week after their first 90 days. Fully remote "
     "arrangements need written approval from the department head and HR, and are reviewed every 12 months."),
    ("Core hours", "Teams keep core hours from 10:00 to 15:00 in the team's main time zone. Outside core hours, "
     "employees set their own schedule but must update their Slack status and calendar."),
    ("Equipment and stipend", "The company provides a laptop, and remote employees receive a one-time 300 USD home "
     "office stipend (shared with the expense policy) and 40 USD per month towards internet costs. Equipment "
     "remains company property and is returned on departure."),
    ("Working from abroad", "Employees may work from another country for up to 30 days per calendar year with manager "
     "approval. Longer stays need a tax and compliance review by HR at least 60 days before travel. Work from "
     "sanctioned countries is not permitted."),
    ("Security", "Remote work follows the IT Security and Access Policy: VPN for production access, no public "
     "Wi-Fi without the VPN, screen lock when away from the device, and no customer data on personal devices."),
    ("Coworking spaces", "A coworking desk can be expensed up to 250 USD per month with manager approval when the "
     "employee has no suitable home workspace. Day passes for team meetups are covered by the team budget."),
]
styles = getSampleStyleSheet()
story = []
for title, body in SECTIONS:
    story.append(Paragraph(title, styles["Title"] if body is None else styles["Heading2"]))
    if body:
        story.append(Paragraph(body, styles["BodyText"]))
    story.append(Spacer(1, 10))
SimpleDocTemplate(str(OUT), pagesize=A4, title="Remote Work Policy").build(story)
print("wrote", OUT)
