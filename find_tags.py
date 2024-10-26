import requests
import xml.etree.ElementTree as ET

# Define SEC parameters and User-Agent
cik = '0000320193'  # Apple Inc. CIK
filing_type = '10-K'
user_agent = 'useragent@email.com'

# Step 1: Retrieve recent filings for the given CIK
search_url = f'https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json'
headers = {'User-Agent': user_agent}
response = requests.get(search_url, headers=headers)
response.raise_for_status()
data = response.json()

# Step 2: Locate the most recent 10-K filing
for form, accession in zip(data['filings']['recent']['form'], data['filings']['recent']['accessionNumber']):
    if form == filing_type:
        accession_number = accession.replace('-', '')
        break

# Step 3: Access the filing directory via `index.json`
base_url = f'https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}'
index_url = f'{base_url}/index.json'
response = requests.get(index_url, headers=headers)
response.raise_for_status()
index_data = response.json()

# Step 4: Broaden search to locate an XML instance document with relevant data
xbrl_url = None
for file in index_data['directory']['item']:
    # Filter for files likely to be main instance documents
    if file['name'].endswith('.xml') and 'cal' not in file['name'] and 'def' not in file['name'] and 'lab' not in file['name'] and 'pre' not in file['name']:
        xbrl_url = f"{base_url}/{file['name']}"
        break

if xbrl_url:
    # Step 5: Download and parse the XBRL instance document
    response = requests.get(xbrl_url, headers=headers)
    response.raise_for_status()
    root = ET.fromstring(response.content)

    # Collect and format tags by namespaces
    print(f"XBRL Tags for Filing (CIK: {cik}, Filing Type: {filing_type}):\n{'-'*50}")
    tags = {}
    for element in root.iter():
        namespace = element.tag.split('}')[0].strip('{')  # Extract namespace
        tag_name = element.tag.split('}')[1] if '}' in element.tag else element.tag
        if namespace not in tags:
            tags[namespace] = []
        tags[namespace].append(tag_name)
    
    # Print tags grouped by namespace
    for namespace, tag_names in tags.items():
        print(f"Namespace: {namespace}")
        for tag_name in sorted(set(tag_names)):  # Deduplicate tags within namespace
            print(f"  - {tag_name}")
        print("\n" + "-"*50)

else:
    print("No primary XBRL instance document found.")
