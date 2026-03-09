"""
Communication Simulation Module
=================================
Generates borrower-facing messages (SMS / Email / Chatbot) using templates
and logs the simulated interactions to the SQLite database.
"""

import os
import datetime
import uuid
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")

# ────────────────────────────── Message Templates ──────────────────────────

TEMPLATES = {
    "SMS reminder": (
        "Hi {name}, this is a friendly reminder that your payment of "
        "₹{amount} is overdue by {days} days.  Please pay at your earliest "
        "convenience.  Ref: {ref}"
    ),
    "Call-center follow-up": (
        "Subject: Payment Follow-Up – Account {ref}\n\n"
        "Dear {name},\n\n"
        "We noticed your account has an outstanding balance of ₹{amount}.  "
        "Our team will reach out shortly to discuss a suitable repayment plan.\n\n"
        "Regards,\nCollections Team"
    ),
    "Field collection escalation": (
        "CHATBOT: Hello {name}, your account {ref} is significantly overdue "
        "(₹{amount}, {days} days).  We'd like to help you find a resolution.  "
        "Reply YES to schedule a call with an advisor."
    ),
}


def generate_messages(
    actions_path: str | None = None,
    raw_path: str | None = None,
) -> pd.DataFrame:
    """Create a message for each borrower based on their recommended action."""
    if actions_path is None:
        actions_path = os.path.join(
            BASE_DIR, "data", "processed", "collection_actions.csv"
        )
    if raw_path is None:
        raw_path = os.path.join(BASE_DIR, "data", "raw", "borrowers.csv")

    actions = pd.read_csv(actions_path)
    raw = pd.read_csv(raw_path)[
        ["borrower_id", "outstanding_balance", "days_past_due"]
    ]
    df = actions.merge(raw, on="borrower_id", how="left")

    records = []
    for _, row in df.iterrows():
        tmpl = TEMPLATES.get(row["recommended_action"], TEMPLATES["SMS reminder"])
        msg = tmpl.format(
            name=row["borrower_id"],
            amount=f"{row.get('outstanding_balance', 0):,.2f}",
            days=int(row.get("days_past_due", 0)),
            ref=row["borrower_id"],
        )
        channel = {
            "SMS reminder": "SMS",
            "Call-center follow-up": "Email",
            "Field collection escalation": "Chatbot",
        }.get(row["recommended_action"], "SMS")

        records.append(
            {
                "log_id": str(uuid.uuid4()),
                "borrower_id": row["borrower_id"],
                "channel": channel,
                "message": msg,
                "timestamp": datetime.datetime.now().isoformat(),
                "status": "sent",
            }
        )

    return pd.DataFrame(records)


def save_communication_logs(output_dir: str | None = None) -> str:
    df = generate_messages()
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "communication_logs.csv")
    df.to_csv(path, index=False)
    print(f"[communication] Generated {len(df)} messages -> {path}")
    return path


if __name__ == "__main__":
    save_communication_logs()
