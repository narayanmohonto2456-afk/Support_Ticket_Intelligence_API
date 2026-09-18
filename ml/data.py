from itertools import product
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "generated_tickets.csv"

CATEGORY_TEMPLATES = {
    "account_access": [
        "I cannot sign in to my {system} account after several attempts",
        "My {system} account is locked and I need access",
        "The password reset link for {system} is not working",
        "Two factor authentication prevents me from accessing {system}",
        "My login credentials are rejected by {system}",
    ],
    "billing": [
        "I was charged twice for my {system} subscription",
        "The latest {system} invoice shows an incorrect amount",
        "My payment for {system} failed but money was deducted",
        "I need a refund for the cancelled {system} service",
        "The billing address on my {system} invoice is incorrect",
    ],
    "technical_issue": [
        "The {system} application crashes whenever I upload a file",
        "I receive a server error while opening the {system} dashboard",
        "The {system} page is blank after the latest update",
        "Data is not loading in {system} even after refreshing",
        "The {system} integration stopped synchronizing records",
    ],
    "feature_request": [
        "Please add dark mode to the {system} application",
        "It would be useful to export {system} reports to PDF",
        "Can you add bulk editing to the {system} dashboard",
        "Please support scheduled reports in {system}",
        "I would like a mobile notification option for {system}",
    ],
}

SYSTEMS = ["employee portal", "customer dashboard", "mobile app", "reporting system"]

PRIORITY_PHRASES = {
    "low": [
        "This is only a suggestion and there is no immediate deadline.",
        "Please consider this improvement for a future release.",
    ],
    "medium": [
        "This affects my regular work but I have a temporary workaround.",
        "Please resolve this during normal business hours.",
    ],
    "high": [
        "This blocks an important task for my team today.",
        "Several users are affected and work cannot continue normally.",
    ],
    "critical": [
        "Production is unavailable for all users and requires immediate action.",
        "This is a security-related outage affecting the entire organization.",
    ],
}


def generate_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Create a deterministic synthetic learning dataset with balanced labels."""
    rows: list[dict[str, str]] = []
    for category, templates in CATEGORY_TEMPLATES.items():
        for template, system, (priority, phrases) in product(
            templates, SYSTEMS, PRIORITY_PHRASES.items()
        ):
            phrase_index = (len(rows) + len(system)) % len(phrases)
            rows.append(
                {
                    "text": f"{template.format(system=system)}. {phrases[phrase_index]}",
                    "category": category,
                    "priority": priority,
                }
            )
    dataframe = pd.DataFrame(rows).drop_duplicates(subset=["text"])
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    return dataframe


if __name__ == "__main__":
    frame = generate_dataset()
    print(f"Generated {len(frame)} tickets at {DATA_PATH}")

