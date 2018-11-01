import json
import requests

filename = "all_organization_data.json"

with open(filename) as f:
	data = json.load(f)

keys = []

for e in data["value"]:
	keys.append(e["WebsiteKey"])
	print(e["Name"])

print()
print("Number of organizations:", len(keys))

# r = requests.get("https://maroonlink.tamu.edu/api/discovery/organization/bykey/Aco")

# print(r.json()["description"])
# f = open("Aco.json", "w")
# f.write(r.text)

# print(keys[3])

baseUrl = "https://maroonlink.tamu.edu/api/discovery/organization/bykey/"

for k in keys:
	f_name = k + ".json"
	url = baseUrl + k
	r = requests.get(url)
	f = open(f_name, "w")
	f.write(r.text)