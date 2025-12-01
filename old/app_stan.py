import requests as rq


# call the API
base_url = "https://api.tomtom.com/search/2/search/"
params = {
    "minFuzzyLevel": 1,
    "maxFuzzyLevel": 2,
    "view": "Unified",
    "relatedPois": "off",
    "key": key,
}

response = rq.get(base_url, params=params, timeout=10)
response.raise_for_status()
json_response = response.json() # save data in json_response

print(response.url)