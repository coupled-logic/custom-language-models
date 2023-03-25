import json

# Create an empty list to hold the dictionaries
merged_data = []

# Read each file and append the dictionaries to the list
# for filename in ['file1.json', 'file2.json', 'file3.json', 'file4.json']:
for filename in ['artisticstyles.json', 'examples.json', 'photographystyles.json', 'weights.json']:
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        merged_data.extend(data)

# Write the merged list to a new file
with open('merged_data.json', 'w') as f:
    json.dump(merged_data, f)