# Serenade: Low-Latency Session-Based Recommendations
<img src="https://icai.ai/wp-content/uploads/2020/01/AIRLabAmsterdam-10-6-gecomprimeerd-transparant.png" width="250" align="right">
This repository contains the official code for session-based recommender system Serenade.
It learns users' preferences by capturing the short-term and sequential patterns from the evolution of user
behaviors and predicts interesting next items with low latency.

# Quick guide: getting started with Serenade.

## Table of contents
1. [Find the best hyperparameter values](#find-hyperparams)
2. [Configure Serenade to use the hyperparameter values](#update-config)
3. [Start the Serenade service](#start-service)
4. [Retrieve recommendations using python](#retrieve-recommendations)
5. [Research experiments](#research-experiments)

This guide assumes you have the following:
- Binary executables for your platform (mac, linux and windows are supported):
```
serving
hyperparameter_search
```
- A configuration file `Default.toml` (an example can be found at `config/_Default.toml`);
- A csv file with training data `retailrocket9_train.txt` in `/datasets/`.
- A csv file with test data `retailrocket9_test.txt` in `/datasets/`.

### Find the best hyperparameter values <a name="find-hyperparams"></a>
We explored different possibilities for optimizing the hyperparameters, and the best performance was obtained using the Tree-Structured Parzen Estimator (TPE), for which we give instructions in the following subsection.

#### Tree-Structured Parzen Estimator (TPE)

* TPE achieves low validation errors when compared to Exhaustive Grid Search ([Bergstra et al](http://proceedings.mlr.press/v28/bergstra13.pdf)).
* It has [an implementation available in Rust (link)](https://docs.rs/tpe).
* The config file still does not handle the hyperparameters' values, so, for now, please change the values in lines 23-25 of `src/bin/tpe_hyperparameter_optm.rs`. For the other config parameters, please edit/review the config file (`config/_Default.toml`).
* Running example:
```bash
./target/<compile_dir>/tpe_hyperparameter_optm config/_Default.toml 
```

The results will be printed out in the terminal, for example:
```
...
Best n_most_recent_sessions: 500
Best neighborhood_size_k: 1000
Best last_items_in_session: 5
Business logic were disabled.
Best value for the goal metric: 0.16249138818029166
```
and also in the output file defined in the config file, for example:
```
out_path = "results.csv"
```

### Configure Serenade to use the hyperparameter values <a name="update-config"></a>
We now update the configuration file `Default.toml` to use the hyperparameter values and set the training_data_path with the location of the retailrocket9_train.txt.
This is an example of the full configuration file
```
config_type = "toml"

[server]
host = "0.0.0.0"
port = 8080
num_workers = 4

[log]
level = "info" # not implemented

[data]
training_data_path=/datasets/retailrocket9_train.txt

[model]
neighborhood_size_k = 1000
max_items_in_session = 7 
m_most_recent_sessions = 250
num_items_to_recommend = 21

[logic]
enable_business_logic = "false"

[hyperparam]
training_data_path = "datasets/retailrocket9_train.txt"
test_data_path = "datasets/retailrocket9_test.txt"
num_iterations = 12
save_records = true
out_path = "results.csv"
enable_business_logic = false
# these configs below are not working yet
n_most_recent_sessions_choices = [100, 500, 1000, 2500]
neighborhood_size_k_choices = [50, 100, 500, 1000, 1500]
last_items_in_session_choices = [1, 2, 3, 5, 7, 10]
```

### Start the Serenade service <a name="start-service"></a>
Start the `serving` binary for your platform with the location of the configuration file `Default.toml` as argument
```bash
./serving Default.toml
```

You can open your webbrowser and goto http://localhost:8080/ you should see an internal page of Serenade.


### Retrieve recommendations using python <a name="retrieve-recommendations"></a>

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



# Citation
> Serenade - Low-Latency Session-Based Recommendation in e-Commerce at Scale

    @article{Kersbergen2022SerenadeScale,
        title = {{Serenade - Low-Latency Session-Based Recommendation in e-Commerce at Scale}},
        year = {2022},
        journal = {SIGMOD},
        author = {Kersbergen, Barrie and Sprangers, Olivier and Schelter, Sebastian}
    }


# License
This project is licensed under the terms of the [Apache 2.0 license](LICENSE).

