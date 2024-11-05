import requests
obj = requests.get('http://api.conceptnet.io/c/en/apple').json()
print(obj.keys())
print(obj['edges'][0])
exit()