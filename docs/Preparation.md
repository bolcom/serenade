Preparation
===

Get and Format Data
---

### How to get the data

Example public datasets: <https://drive.google.com/file/d/1zeGdSoxfkRW3oAvpYGv9u2AGHP-jghfp/view?usp=sharing>

Download and extract the data files.

### Data format

Serenade needs training data for generating recommendations and test data to evaluate the quality of your recommendations.
It supports binary AVRO files and CSV text files as input. 
If your data has less than 100 million records it is advised to use CSV files to get started with Serenade. A machine with 16GB of RAM memory is advised.

A CSV file must be TAB (\t) separated and look something like this:
```
SessionId       ItemId  Time
10036   14957   1592337718.0
10036   14713   1592337765.0
10036   2625    1592338184.0
10037   7267    1591979344.0
10037   13892   1591979380.0
10037   7267    1591979504.0
```

* **SessionId** A unique session identifier (unsigned 32 bit integers supported)
* **ItemId** an identifier for a product or item that a visitor interacted with. (unsigned 64 bit integers supported)
* **Time** epoch in seconds. (32 bit float and unsigned 32 bit integers supported)


Configure Application
---

All application configuration parameters can be set with a single config file in TOML format. Some parameters (currently only `training_data_path` and `num_workers`) can alternatively be set via environment variables.

### Minimum Configuration

The only parameter value that is required and has no default value to fall back on is `training_data_path`. The user must have a training data file, with appropriate data formatting (see above). Since this parameter can be set with an environment variable, the config file can be left out entirely. In this case, the default values specified in [CONFIG](CONFIG.md) are used for all remaining parameters.

See [CONFIG](CONFIG.md) for further configuration details.
