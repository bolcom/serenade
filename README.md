# Serenade: Low-Latency Session-Based Recommendations
<img src="https://icai.ai/wp-content/uploads/2020/01/AIRLabAmsterdam-10-6-gecomprimeerd-transparant.png" width="250" align="right">
This repository contains the official code for session-based recommender system Serenade, which employs VMIS-kNN. 
It learns users' preferences by capturing the short-term and sequential patterns from the evolution of user behaviors and predicts interesting next items with low latency with support for millions of distinct items.
VMIS-kNN is an index-based variant of a state-of-the-art nearest neighbor algorithm to session-based recommendation, which scales to use cases with hundreds of millions of clicks to search through.

# Quick guide: getting started with Serenade.

## Table of contents
1. [Downloads](#downloads)
2. [Train- and testset](#dataset)
3. [Find the best hyperparameter values](#find-hyperparams)
4. [Configure Serenade to use the hyperparameter values](#update-config)
5. [Start the Serenade service](#start-service)
6. [Retrieve recommendations using python](#retrieve-recommendations)
7. [Research experiments](#research-experiments)

### Downloads <a name="downloads"></a>
Serenade can be downloaded [here](https://github.com/bolcom/serenade/releases). Binary executables are available for Windows, Linux and MacOS.
We provide a sample config file for Serenade [here](https://github.com/bolcom/serenade/blob/main/config/_Default.toml)

Extract the just downloaded archive. You now have the following files.
```
serving
tpe_hyperparameter_optm
```
- A configuration file `Default.toml` (an example can be found at `config/_Default.toml`);
- A csv file with training data `train.txt` in `/datasets/`. Which we describe in the next section.
- A csv file with test data `test.txt` in `/datasets/`.


### Train- and testset <a name="dataset"></a>
A train- and testset must be created from historical user-item click data, outside of Serenade. Each row in the training- or test set should contain an historical user-item interaction event with the following fields:
* ```SessionId``` the ID of the session. Format: 64 bit Integer
* ```ItemId``` the ID of the interacted item. Format: 64 bit Integer
* ```Time``` the time when the user-item interaction occurred. In epoch seconds: 32 bit Floating point.

The last 24 hours in the historical data can be used as test-set while the rest of the sessions can be used as the training-set and written as plain text using a ```'\t'``` as field separator.
This is an example of a training data CSV file train.txt:
```
SessionId	ItemId	Time
77	453279	1434846304
77	321706	1434846392
77	257070	1434846434
85	52383	1434987517
```

This is an example of test data test.txt:
```
SessionId	ItemId	Time
897	91755	1435330427
897	91755	1435330595
1011	387377	1435296520
```


### Find the best hyperparameter values <a name="find-hyperparams"></a>
The next step is finding the hyperparameters for the train and test-datasets. 
Serenade uses Tree-Structured Parzen Estimator (TPE) for finding the hyperparameters. TPE achieves low validation errors compared to Exhaustive Grid Search ([Bergstra et al](http://proceedings.mlr.press/v28/bergstra13.pdf)).

* The hyperparameter search can be started using:
```bash
./tpe_hyperparameter_optm config/_Default.toml 
```
note: At this moment the hyper-parameter value ranges in the config file are ignored and Serenade uses the hardcoded ranges defined in lines 23-25 of `src/bin/tpe_hyperparameter_optm.rs`. The hardcoded ranges are commonly used and should give good results. 

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
We now update the configuration file `Default.toml` to use the hyperparameter values and set the training_data_path with the location of the ```train.txt```.
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
training_data_path=/datasets/train.txt

[model]
neighborhood_size_k = 1000
max_items_in_session = 7 
m_most_recent_sessions = 250
num_items_to_recommend = 21

[logic]
enable_business_logic = "false"

[hyperparam]
training_data_path = "datasets/train.txt"
test_data_path = "datasets/test.txt"
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
> [Serenade - Low-Latency Session-Based Recommendation in e-Commerce at Scale](https://ssc.io/pdf/modds003.pdf)

    @article{Kersbergen2022SerenadeScale,
        title = {{Serenade - Low-Latency Session-Based Recommendation in e-Commerce at Scale}},
        year = {2022},
        journal = {SIGMOD},
        author = {Kersbergen, Barrie and Sprangers, Olivier and Schelter, Sebastian}
    }


# License
This project is licensed under the terms of the [Apache 2.0 license](LICENSE.md).

