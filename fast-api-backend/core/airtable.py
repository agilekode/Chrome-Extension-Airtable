import requests
from config import AIRTABLE_BASE_ID, AIRTABLE_API_KEY, ENDORSEMENTS_TABLE_ID

class AirtableClient:
    def __init__(self):
        self.base_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}"
        self.headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}

    def fetch_endorsement_pdf(self, endorsement_id: str):
        url = f"{self.base_url}/{ENDORSEMENTS_TABLE_ID}/{endorsement_id}"
        resp = requests.get(url, headers=self.headers)
        resp.raise_for_status()
        fields = resp.json().get("fields", {})
        return fields.get("Endorsement PDF", [])
