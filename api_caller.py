import requests
json_req ={'sales1': 245, 'sales2': 198, 'sales3': 431, 'sales4': 432,'deman_level':'high','best_seller':'A' }
url = 'http://127.0.0.1:5000/pred'
# # create the post request
req = requests.post(url,json=json_req)
print(req.status_code)
# print the requested result
print(req.json())