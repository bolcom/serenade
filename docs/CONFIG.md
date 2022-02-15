Application Configuration
===

All application configuration parameters can be set with a single config file in TOML format. Some parameters (currently only `training_data_path` and `num_workers`) can alternatively be set via environment variables.

Minimum Configuration
---

The only parameter value that is required and has no default value to fall back on is `training_data_path`. The user must have a training data file, with appropriate data formatting (see [DATA](DATA.md)). Since this parameter can be set with an environment variable, the config file can be left out entirely. In this case, the default values specified below are used for all remaining parameters.

Full Parameter List
---

| Config Section | Parameter | Type | Description | Required | Default | Sources |
| --- | --- | --- | --- | --- | --- | --- |
| `data` | `training_data_path` | str | Path to training data file | :heavy_check_mark: | | Config file or environment variable |
| `server` | `num_workers` | int | Number of server worker threads | | Number of CPUs detected | Config file or environment variable |
| `server` | `host` | str | Host at which server should listen | | `"0.0.0.0"` | Config file |
| `server` | `port` | int | Port at which server should listen | | `8080` | Config file |
| `log` | `level` | str | Server logging level (not implemented) | | | Config file |
| `model` | `sample_size_m` | int | <!-- TODO --> | | `500` | Config file |
| `model` | `neighborhood_size_k` | int | Number of similar sessions to compare to current session | | `500` | Config file |
| `model` | `num_items_to_recommend` | int | Number of predictions the model should make | | `21` | Config file |
| `model` | `max_items_in_session` | int | Size of current session history to consider as model input | | `2` | Config file |

Example
---

Example configuration file specifying all of the above parameters:

```toml
[server]
num_workers = 4
host = "0.0.0.0"
port = 8080

[log]
level = "info"  # not implemented

[data]
training_data_path = "/path/to/training/data"

[model]
sample_size_m = 500
neighborhood_size_k = 500
num_items_to_recommend = 21
max_items_in_session = 2
```
