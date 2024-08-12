import os
from dotenv import load_dotenv
import requests

def scrape_linkedin_profile(linkedin_url: str):

    load_dotenv()

    # This code is used to get Profile data
    # But since we have limited access to the API, we will comment this coide and rather use already downloaded data from gist. Please see notepad for more details
    # Also code was not executed because of the limited access to the API
    ##################################################################################################################################################
    ##################################################################################################################################################
    ##################################################################################################################################################
    ##################################################################################################################################################

    """ Scrape information from a LinkedIn profile """
    # api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    # headers = {"Authorization": f"Bearer {os.environ.get('PROXYCURL_API_KEY')}"}

    # response = requests.get(f"{api_endpoint}?url={linkedin_url}", headers=headers)

    # data = response.json()

    ##################################################################################################################################################
    ##################################################################################################################################################
    ##################################################################################################################################################
    ##################################################################################################################################################

    gist_response = requests.get("https://gist.githubusercontent.com/ankurmalik1978/253257e0191d3a5fb520a4e933a8c5b7/raw/c3f8df88ed9c5a7b853d7d795f20bc5bb8dea9e2/linkedinprofile.json")
    data = gist_response.json()

    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
        and k not in ["people_also_viewed", "certifications"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data