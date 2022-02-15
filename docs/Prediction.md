Run the Serenade Service and Make Predictions
===




### Interact with Serenade using python
This example uses the retailrocket9_train.txt dataset.
```python
import requests
from requests.exceptions import HTTPError
try:
    myurl = 'http://localhost:8080/v1/recommend'
    params = dict(
        session_id='144',
        user_consent='true',
        item_id='453279',
    )
    response = requests.get(url=myurl, params=params)
    response.raise_for_status()
    # access json content
    jsonResponse = response.json()
    print(jsonResponse)
except HTTPError as http_err:
    print(f'HTTP error occurred: {http_err}')
except Exception as err:
    print(f'Other error occurred: {err}')
```
```
[72916, 84895, 92210, 176166, 379693, 129343, 321706, 257070]
```
The returned json object is a list with recommended items. 