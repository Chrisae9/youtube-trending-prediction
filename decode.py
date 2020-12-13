import json

with open('data/categories.json') as json_file:
    data = json.load(json_file)
    formatted_data = dict()
    for i in data['items']:
        formatted_data[i['snippet']['title']] = int(i['id'])

    print(formatted_data)

    with open('data/formatted_categories.json', 'w') as outfile:
        json.dump(formatted_data, outfile, indent=2)

    