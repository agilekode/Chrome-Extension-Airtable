import requests
from fastapi import APIRouter,HTTPException
from config import AIRTABLE_BASE_ID,AIRTABLE_API_KEY,AIRTABLE_TABLE_NAME,ENDORSEMENT_TABLE_NAME
router = APIRouter()
@router.get("/search_policy")
def search_policy(policy_name: str):
    """
    Search for a record in Airtable by Policy Name and build metadata with base policy + endorsements.
    """
    SEARCH_COLUMN = "Policy Name"
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_NAME}"

    formula = f"{{{SEARCH_COLUMN}}}='{policy_name}'"
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    params = {"filterByFormula": formula, "maxRecords": 1}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    records = response.json().get("records", [])
    if not records:
        return {"message": "No matching records found."}

    record = records[0]
    fields = record.get("fields", {})

    meta_data = {}

    # 1. Get base policy filename
    attachments = fields.get("policy attachments", [])
    if attachments:
        base_filename = attachments[0].get("filename")
        meta_data["base_policy_filename"] = base_filename

    # 2. Loop over endorsements
    endorsement_ids = fields.get("Endorsements", [])
    for idx, endorsement_id in enumerate(endorsement_ids, start=1):
        end_url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{ENDORSEMENT_TABLE_NAME}/{endorsement_id}"
        end_resp = requests.get(end_url, headers=headers)
        if end_resp.status_code != 200:
            continue  # skip if error
        end_record = end_resp.json()
        end_fields = end_record.get("fields", {})

        endorsement_files = end_fields.get("Endorsement PDF", [])
        if endorsement_files:
            filename = endorsement_files[0].get("filename")
            meta_data[f"endorsement_{idx}"] = filename

    return {"policy_name":policy_name ,"metadata":meta_data , "status_code":200}