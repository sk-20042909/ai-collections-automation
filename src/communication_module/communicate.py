"""
Communication Module
====================
Generates personalised outreach messages for each borrower based on their
risk tier and recommended channel.

Templates:
    SMS       – concise payment reminder
    Email     – formal notice with payment link
    Chatbot   – conversational nudge
    Payment Assistance – proactive assistance offer for high / very-high risk
"""

import os
from datetime import datetime
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

TEMPLATES: dict[str, str] = {
    "sms": (
        "Hi {name}, this is a friendly reminder that your credit-card "
        "account ({borrower_id}) has an outstanding balance. "
        "Please make a payment at your earliest convenience. "
        "Reply HELP for assistance."
    ),
    "email": (
        "Dear Valued Customer ({borrower_id}),\n\n"
        "Our records indicate an outstanding balance on your account.  "
        "Please arrange a payment by the due date to avoid late fees.\n\n"
        "If you need help, contact us at support@collections.example.com\n\n"
        "Regards,\nCollections Team"
    ),
    "chatbot": (
        "👋 Hey! Just checking in about your credit-card account "
        "({borrower_id}). Would you like help setting up a payment plan? "
        "Type YES to get started."
    ),
    "payment_assistance": (
        "Dear Customer ({borrower_id}),\n\n"
        "We understand that managing payments can be challenging. "
        "Based on your account profile, you may qualify for a customised "
        "payment plan or financial-hardship programme.\n\n"
        "Please reply to this message or call 1-800-555-0199 to speak "
        "with a dedicated account specialist.\n\n"
        "We're here to help.\n"
        "Collections Support Team"
    ),
}


def _pick_template(row: pd.Series) -> tuple[str, str]:
    tier = row["risk_tier"]
    if tier in ("High Risk", "Very High Risk"):
        return "payment_assistance", TEMPLATES["payment_assistance"]
    channel = str(row.get("channel", "")).lower()
    if "sms" in channel:
        return "sms", TEMPLATES["sms"]
    if "phone" in channel or "email" in channel:
        return "email", TEMPLATES["email"]
    return "chatbot", TEMPLATES["chatbot"]


def generate_messages(strategies_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in strategies_df.iterrows():
        tpl_name, tpl = _pick_template(row)
        message = tpl.format(
            name="Customer",
            borrower_id=row["borrower_id"],
        )
        rows.append({
            "borrower_id": row["borrower_id"],
            "risk_tier": row["risk_tier"],
            "channel": row.get("channel", "Email"),
            "template": tpl_name,
            "message": message,
            "generated_at": datetime.now().isoformat(),
        })
    df = pd.DataFrame(rows)
    out = os.path.join(DATA_DIR, "communication_log.csv")
    df.to_csv(out, index=False)
    print(f"[communication] {len(df)} messages generated -> {out}")
    return df


def run_communication() -> pd.DataFrame:
    strats = pd.read_csv(os.path.join(DATA_DIR, "collection_strategies.csv"))
    return generate_messages(strats)


if __name__ == "__main__":
    run_communication()
