import os
import requests
import json

# Define a safe data directory
data_dir = "data"  # Change to "./data" or full path if needed
os.makedirs(data_dir, exist_ok=True)

# API URL
url = "https://pxweb.asub.ax:443/PXWeb/api/v1/en/Statistik/INLO/IN/IN003.px"

# Payload for JSON response
payload = {
    "query": [
        {
            "code": "kommun",
            "selection": {
                "filter": "item",
                "values": [
                    "035", "043", "060", "062", "065", "076", "170", "295", "318",
                    "417", "438", "736", "766", "771", "941", "478", "MK21"
                ]
            }
        }
    ],
    "response": {"format": "json"}
}

# Fetch data
response = requests.post(url, json=payload)

if response.status_code == 200:
    # Parse JSON response
    data = response.json()

    # Save JSON data to a file
    file_path = os.path.join(data_dir, "data.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    
    print(f"Data fetched and saved as {file_path} successfully!")

    # Optionally, print sample output
    print(json.dumps(data, indent=4))
else:
    print("Error:", response.status_code, response.text)
