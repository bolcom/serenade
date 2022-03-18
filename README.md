# Serenade: Low-Latency Session-Based Recommendations
<img src="https://icai.ai/wp-content/uploads/2020/01/AIRLabAmsterdam-10-6-gecomprimeerd-transparant.png" width="250" align="right">
This repository contains the official code for session-based recommender system Serenade, which employs VMIS-kNN. 
It learns users' preferences by capturing the short-term and sequential patterns from the evolution of user behaviors and predicts interesting next items with low latency with support for millions of distinct items.
VMIS-kNN is an index-based variant of a state-of-the-art nearest neighbor algorithm to session-based recommendation, which scales to use cases with hundreds of millions of clicks to search through.

# Quick guide: getting started with Serenade.

## Table of contents
1. [Downloads](#downloads)
2. [Find the best hyperparameter values](#find-hyperparams)
3. [Configure Serenade to use the hyperparameter values](#update-config)
4. [Start the Serenade service](#start-service)
5. [Retrieve recommendations using python](#retrieve-recommendations)
6. [Using your own train- and testset](#dataset)
7. [Evaluate a testsetl](#evaluator)


### Downloads <a name="downloads"></a>
Serenade can be downloaded [here](https://github.com/bolcom/serenade/releases). Binary executables are available for Windows, Linux and MacOS.
Download the [toy example project](https://github.com/bolcom/serenade/raw/main/assets/example/example.zip) which contains toy datasets and a preconfigured configuration file.

Extract both downloaded files in the same directoy. You now have the following files:
```
serving
tpe_hyperparameter_optm
evaluator
train.txt
test.txt
valid.txt
example.toml
```

### Find the best hyperparameter values <a name="find-hyperparams"></a>
The next step is finding the hyperparameters using the train and test-datasets. 
Serenade uses Tree-Structured Parzen Estimator (TPE) for finding the hyperparameters. TPE achieves low validation errors compared to Exhaustive Grid Search ([Bergstra et al](http://proceedings.mlr.press/v28/bergstra13.pdf)).
The section `[hyperparam]` in the `example.toml` contains the ranges of hyperparameter values that will be explored.

* The hyperparameter search can be started using:
```bash
./tpe_hyperparameter_optm example.toml 
```

The results will be printed out in the terminal, for example:
```
===============================================================
===          HYPER PARAMETER OPTIMIZATION RESULTS          ====
===============================================================
MRR@20 for validation data: 0.3197
MRR@20 for test data: 0.3401
enabled business_logic for evaluation:false
best hyperparameter values:
n_most_recent_sessions:1502
neighborhood_size_k:288
idf_weighting:2
last_items_in_session:4
HPO done
```
and also in the output file defined in the config file, for example:
```
out_path = "results.csv"
```

### Configure Serenade to use the hyperparameter values <a name="update-config"></a>
We now update the `[model]` values in configuration file `example.toml` to use the hyperparameter values and set the training_data_path with the location of the ```train.txt```.
This is the content of the example configuration file with the new `[model]` paramer values.
```
config_type = "toml"

[server]
host = "0.0.0.0"
port = 8080
num_workers = 4

[log]
level = "info" # not implemented

[data]
training_data_path="train.txt"

[model]
m_most_recent_sessions = 1502
neighborhood_size_k = 288
max_items_in_session = 4
num_items_to_recommend = 21
idf_weighting = 1

[logic]
enable_business_logic = "false"

[hyperparam]
training_data_path = "train.txt"
test_data_path = "test.txt"
validation_data_path = "valid.txt"
num_iterations = 15
save_records = true
out_path = "results.csv"
enable_business_logic = false
n_most_recent_sessions_range = [100, 2500]
neighborhood_size_k_range = [50, 1500]
last_items_in_session_range = [1, 20]
idf_weighting_range = [0, 5]
```

### Start the Serenade service <a name="start-service"></a>
Start the `serving` binary for your platform with the location of the configuration file as argument
```bash
./serving example.toml
```


You can open your webbrowser and goto http://localhost:8080/ you should see an internal page of Serenade.

Serenade exposes [Prometheus metrics](http://localhost:8080/internal/prometheus) out-of-the-box for monitoring. 

### Retrieve recommendations using python <a name="retrieve-recommendations"></a>

```python
import requests
from requests.exceptions import HTTPError
try:
    myurl = 'http://localhost:8080/v1/recommend'
    params = dict(
        session_id='144',
        user_consent='true',
        item_id='13598',
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
[2835,10,12068,4313,3097,8028,3545,7812,17519,1164,17935,1277,13335,8655,14664,14556,6868,13509,9248,2498,11724]
```
The returned json object is a list with recommended items.


### Using your own train- and testset <a name="dataset"></a>
A train- and testset must be created from historical user-item click data, outside of Serenade. Each row in the training- or test set should contain an historical user-item interaction event with the following fields:
* ```SessionId``` the ID of the session. Format: 64 bit Integer
* ```ItemId``` the ID of the interacted item. Format: 64 bit Integer
* ```Time``` the time when the user-item interaction occurred. In epoch seconds: 32 bit Floating point.

The last 24 hours in the historical data can be used as test-set while the rest of the sessions can be used as the training-set and written as plain text using a ```'\t'``` as field separator.
This is an example of a training data CSV file train.txt:
```
SessionId       ItemId  Time
10036   14957   1592337718.0
10036   14713   1592337765.0
10036   2625    1592338184.0
10037   7267    1591979344.0
10037   13892   1591979380.0
10037   7267    1591979504.0
10037   3595    1591979784.0
10038   6424    1591008704.0
```

### Evaluate a testset <a name="evaluator"></a>
The `evaluator` application can be used to evaluate a test dataset. It reports on several metrics.
* The evaluation can be started using:
```bash
./evaluator example.toml 
```

```
===============================================================
===               START EVALUATING TEST FILE               ====
===============================================================
Mrr@20,Ndcg@20,HitRate@20,Popularity@20,Precision@20,Coverage@20,Recall@20,F1score@20
0.3277,0.3553,0.6402,0.0499,0.0680,0.2765,0.4456,0.1180
Qty test evaluations: 931
Prediction latency
p90 (microseconds): 66
p95 (microseconds): 66
p99.5 (microseconds): 66
```


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

