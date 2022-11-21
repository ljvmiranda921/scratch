import json
import requests
from bs4 import BeautifulSoup

if __name__ == "__main__":
    # Here's my own karl_gg account
    karl_gg_url = "https://karl.gg/preview/17779#/"
    resp = requests.get(karl_gg_url)
    if resp.ok:
        soup = BeautifulSoup(resp.content, "html.parser")
        data = soup.body.find("loadout-preview-page").attrs
        user = json.loads(data[":loadout-data"])

        print(user.get("creator").get("email"))
