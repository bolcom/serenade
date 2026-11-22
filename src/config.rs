use std::convert::TryInto;
use std::ffi::OsStr;
use std::fs::File;

use justconfig::item::ValueExtractor;
use justconfig::processors::Trim;
use justconfig::sources::env::Env;
use justconfig::sources::text::ConfigText;
use justconfig::ConfPath;
use justconfig::Config;

use crate::config_processors::Unquote;

// Set some default values
const DEFAULT_MOST_RECENT_SESSIONS_M: usize = 500;
const DEFAULT_NEIGHBORHOOD_SIZE_K: usize = 500;
const DEFAULT_NUM_ITEMS_TO_RECOMMEND: usize = 21;
const DEFAULT_MAX_ITEMS_IN_SESSION: usize = 2;
const DEFAULT_IDF_WEIGHTING: f64 = 1.0;

pub struct AppConfig {
    pub server: ServerConfig,
    pub log: LogConfig,
    pub data: DataConfig,
    pub model: ModelConfig,
    pub logic: LogicConfig,
    pub hyperparam: HyperparamConfig,
}

pub struct ServerConfig {
    pub host: String,
    pub port: usize,
    pub num_workers: usize,
}

pub struct LogConfig {
    pub level: String,
}

pub struct DataConfig {
    pub training_data_path: String,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ModelConfig {
    pub knn: crate::vmisknn::algorithms::KnnConfig,
    pub m_most_recent_sessions: usize,
    pub neighborhood_size_k: usize,
    pub num_items_to_recommend: usize,
    pub max_items_in_session: usize,
    pub idf_weighting: f64,
}

pub struct LogicConfig {
    pub enable_business_logic: bool,
    pub example_item_id: usize,
}

pub struct HyperparamConfig {
    pub training_data_path: String,
    pub test_data_path: String,
    pub validation_data_path: String,
    pub num_iterations: usize,
    pub seed: u64,
    pub search_ranges: std::collections::BTreeMap<String, String>,
    pub save_records: bool,
    pub out_path: String,
    pub enable_business_logic: bool,
    pub n_most_recent_sessions_range: String,
    pub neighborhood_size_k_range: String,
    pub last_items_in_session_range: String,
    pub idf_weighting_range: String,
}

impl AppConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.server.host.trim().is_empty(),
            "server.host must not be empty"
        );
        anyhow::ensure!(
            (1..=65535).contains(&self.server.port),
            "server.port must be in 1..=65535"
        );
        anyhow::ensure!(
            self.model.idf_weighting.is_finite(),
            "model.idf_weighting must be finite"
        );
        for (name, value) in [
            ("server.num_workers", self.server.num_workers),
            (
                "model.m_most_recent_sessions",
                self.model.m_most_recent_sessions,
            ),
            ("model.neighborhood_size_k", self.model.neighborhood_size_k),
            (
                "model.num_items_to_recommend",
                self.model.num_items_to_recommend,
            ),
            (
                "model.max_items_in_session",
                self.model.max_items_in_session,
            ),
        ] {
            anyhow::ensure!(value > 0, "{} must be positive", name);
        }
        Ok(())
    }

    pub fn new(config_path: String) -> AppConfig {
        Self::try_new(config_path).expect("load configuration")
    }

    pub fn try_new(config_path: String) -> anyhow::Result<AppConfig> {
        // Initialize config object
        let mut conf = Config::default();

        // Check if there is a config file
        if !config_path.is_empty() {
            let config_file = File::open(&config_path)
                .map_err(|e| anyhow::anyhow!("open configuration {}: {}", config_path, e))?;
            let config_text = ConfigText::new(config_file, &config_path)
                .map_err(|e| anyhow::anyhow!("parse configuration {}: {}", config_path, e))?;
            conf.add_source(config_text);
        }

        // Define config params from environment variables
        let config_env = Env::new(&[
            (
                ConfPath::from(&["data", "training_data_path"]),
                OsStr::new("TRAINING_DATA"),
            ),
            (
                ConfPath::from(&["server", "num_workers"]),
                OsStr::new("NUM_WORKERS"),
            ),
        ]);
        conf.add_source(config_env);

        // Parse into custom config struct
        let config = AppConfig::parse(conf)?;
        config.validate()?;
        Ok(config)
    }

    fn parse(conf: justconfig::Config) -> anyhow::Result<AppConfig> {
        Ok(AppConfig {
            server: ServerConfig::parse(&conf, ConfPath::from(&["server"]))?,
            log: LogConfig::parse(&conf, ConfPath::from(&["log"]))?,
            data: DataConfig::parse(&conf, ConfPath::from(&["data"]))?,
            model: ModelConfig::parse(&conf, ConfPath::from(&["model"]))?,
            logic: LogicConfig::parse(&conf, ConfPath::from(&["logic"]))?,
            hyperparam: HyperparamConfig::parse(&conf, ConfPath::from(&["hyperparam"]))?,
        })
    }
}

impl ServerConfig {
    fn parse(conf: &Config, path: ConfPath) -> anyhow::Result<ServerConfig> {
        Ok(ServerConfig {
            host: conf
                .get(path.push("host"))
                .unquote()
                .value_or_default("host", String::from("0.0.0.0"))?,
            port: conf
                .get(path.push("port"))
                .trim()
                .value_or_default("port", 8080)?,
            num_workers: conf.get(path.push("num_workers")).trim().value_or_default(
                "num_workers",
                sys_info::cpu_num().unwrap_or(1).try_into().unwrap_or(1),
            )?,
        })
    }
}

impl LogConfig {
    fn parse(conf: &Config, path: ConfPath) -> anyhow::Result<LogConfig> {
        Ok(LogConfig {
            level: conf
                .get(path.push("level"))
                .unquote()
                .value_or_default("level", String::new())?,
        })
    }
}

impl DataConfig {
    fn parse(conf: &Config, path: ConfPath) -> anyhow::Result<DataConfig> {
        Ok(DataConfig {
            training_data_path: conf
                .get(path.push("training_data_path"))
                .unquote()
                .value()
                .map_err(|e| anyhow::anyhow!("{}", e))?,
        })
    }
}

impl ModelConfig {
    fn parse(conf: &Config, path: ConfPath) -> anyhow::Result<ModelConfig> {
        Ok(ModelConfig {
            knn: parse_knn_config(conf, &path)?,
            m_most_recent_sessions: conf
                .get(path.push("m_most_recent_sessions"))
                .trim()
                .value_or_default("m_most_recent_sessions", DEFAULT_MOST_RECENT_SESSIONS_M)?,
            neighborhood_size_k: conf
                .get(path.push("neighborhood_size_k"))
                .trim()
                .value_or_default("neighborhood_size_k", DEFAULT_NEIGHBORHOOD_SIZE_K)?,
            num_items_to_recommend: conf
                .get(path.push("num_items_to_recommend"))
                .trim()
                .value_or_default("num_items_to_recommend", DEFAULT_NUM_ITEMS_TO_RECOMMEND)?,
            max_items_in_session: conf
                .get(path.push("max_items_in_session"))
                .trim()
                .value_or_default("max_items_in_session", DEFAULT_MAX_ITEMS_IN_SESSION)?,
            idf_weighting: conf
                .get(path.push("idf_weighting"))
                .trim()
                .value_or_default("idf_weighting", DEFAULT_IDF_WEIGHTING)?,
        })
    }
}

impl LogicConfig {
    fn parse(conf: &Config, path: ConfPath) -> anyhow::Result<LogicConfig> {
        Ok(LogicConfig {
            enable_business_logic: conf
                .get(path.push("enable_business_logic"))
                .trim()
                .value()
                .map_err(|e| anyhow::anyhow!("{}", e))?,
            example_item_id: conf
                .get(path.push("example_item_id"))
                .trim()
                .value_or_default("example_item_id", 0)?,
        })
    }
}

impl HyperparamConfig {
    fn parse(conf: &Config, path: ConfPath) -> anyhow::Result<HyperparamConfig> {
        let mut search_ranges = std::collections::BTreeMap::new();
        for key in [
            "lambda_spw",
            "lambda_snh",
            "lambda_inh",
            "lambda_ipw",
            "lambda_idf",
            "pop_boost",
        ] {
            if let Some(value) = knn_value(conf, &path, &format!("{}_range", key))? {
                search_ranges.insert(key.to_owned(), value);
            }
        }
        Ok(HyperparamConfig {
            seed: conf
                .get(path.push("seed"))
                .trim()
                .value_or_default("seed", 42)?,
            search_ranges,
            training_data_path: conf
                .get(path.push("training_data_path"))
                .unquote()
                .value_or_default("training_data_path", String::new())?,
            test_data_path: conf
                .get(path.push("test_data_path"))
                .unquote()
                .value_or_default("test_data_path", String::new())?,
            validation_data_path: conf
                .get(path.push("validation_data_path"))
                .unquote()
                .value_or_default("validation_data_path", String::new())?,
            num_iterations: conf
                .get(path.push("num_iterations"))
                .trim()
                .value_or_default("num_iterations", 15)?,
            save_records: conf
                .get(path.push("save_records"))
                .trim()
                .value_or_default("save_records", true)?,
            out_path: conf
                .get(path.push("out_path"))
                .unquote()
                .value_or_default("out_path", String::from("results.csv"))?,
            enable_business_logic: conf
                .get(path.push("enable_business_logic"))
                .trim()
                .value_or_default("enable_business_logic", true)?,
            n_most_recent_sessions_range: conf
                .get(path.push("n_most_recent_sessions_range"))
                .trim()
                .value_or_default("n_most_recent_sessions_range", String::from("[100, 2500]"))?,
            neighborhood_size_k_range: conf
                .get(path.push("neighborhood_size_k_range"))
                .trim()
                .value_or_default("neighborhood_size_k_range", String::from("[50, 1500]"))?,
            last_items_in_session_range: conf
                .get(path.push("last_items_in_session_range"))
                .trim()
                .value_or_default("last_items_in_session_range", String::from("[1, 20]"))?,
            idf_weighting_range: conf
                .get(path.push("idf_weighting_range"))
                .trim()
                .value_or_default("idf_weighting_range", String::from("[0, 5]"))?,
        })
    }
}

trait OptionalValue {
    fn value_or_default<T: std::str::FromStr>(self, key: &str, default: T) -> anyhow::Result<T>
    where
        T::Err: std::error::Error + 'static;
}
impl OptionalValue for Result<justconfig::item::StringItem, justconfig::error::ConfigError> {
    fn value_or_default<T: std::str::FromStr>(self, key: &str, default: T) -> anyhow::Result<T>
    where
        T::Err: std::error::Error + 'static,
    {
        match self.value() {
            Ok(value) => Ok(value),
            Err(justconfig::error::ConfigError::ValueNotFound(_)) => Ok(default),
            Err(error) => Err(anyhow::anyhow!("{}: {}", key, error)),
        }
    }
}

fn knn_value(conf: &Config, path: &ConfPath, key: &str) -> anyhow::Result<Option<String>> {
    let value: Result<String, _> = conf.get(path.push(key)).unquote().trim().value();
    match value {
        Ok(value) => Ok(Some(value)),
        Err(justconfig::error::ConfigError::ValueNotFound(_)) => Ok(None),
        Err(error) => Err(anyhow::anyhow!("Invalid model.knn.{}: {}", key, error)),
    }
}
fn parse_knn_config(
    conf: &Config,
    path: &ConfPath,
) -> anyhow::Result<crate::vmisknn::algorithms::KnnConfig> {
    use crate::vmisknn::algorithms::{Algorithm, KnnConfig};
    let algorithm = knn_value(conf, path, "algorithm")?
        .map(|s| {
            s.parse::<Algorithm>()
                .map_err(|e| anyhow::anyhow!("Invalid model.algorithm: {}", e))
        })
        .transpose()?
        .unwrap_or(Algorithm::Vmis);
    let mut config = KnnConfig::for_algorithm(algorithm);
    let options = path.push("knn");
    // These Python experiments require online mutation, click-time state or a
    // separate catalog postprocessor. Reject explicit requests rather than ignore them.
    for key in [
        "dwelling_time",
        "push_reminders",
        "add_reminders",
        "extend",
        "last_n_days",
        "last_n_clicks",
        "sampling",
    ] {
        anyhow::ensure!(
            knn_value(conf, &options, key)?.is_none(),
            "model.knn.{} is not supported; see docs/knn-compatibility.md",
            key
        );
    }
    macro_rules! option {
        ($field:ident) => {
            if let Some(value) = knn_value(conf, &options, stringify!($field))? {
                config.$field = value
                    .parse()
                    .map_err(|_| anyhow::anyhow!("Invalid model.knn.{}", stringify!($field)))?;
            }
        };
    }
    option!(similarity);
    option!(weighting);
    option!(weighting_score);
    option!(weighting_time);
    option!(idf_weighting);
    option!(idf_weighting_session);
    option!(remind);
    option!(normalize);
    option!(pop_boost);
    macro_rules! decay {
        ($field:ident) => {
            if let Some(value) = knn_value(conf, &options, stringify!($field))? {
                config.$field =
                    if value == "none" {
                        None
                    } else {
                        Some(value.parse().map_err(|_| {
                            anyhow::anyhow!("Invalid model.knn.{}", stringify!($field))
                        })?)
                    };
            }
        };
    }
    decay!(lambda_spw);
    decay!(lambda_snh);
    decay!(lambda_inh);
    decay!(lambda_ipw);
    decay!(lambda_idf);
    config
        .validate()
        .map_err(|e| anyhow::anyhow!("Invalid kNN scoring configuration: {}", e))?;
    Ok(config)
}

#[cfg(test)]
mod knn_config_tests {
    use super::*;
    use crate::vmisknn::algorithms::Algorithm;
    fn model(text: &str) -> ModelConfig {
        let mut config = Config::default();
        config.add_source(ConfigText::new(std::io::Cursor::new(text.as_bytes()), "test").unwrap());
        ModelConfig::parse(&config, ConfPath::from(&["model"])).unwrap()
    }
    #[test]
    fn old_configuration_defaults_to_legacy_vmis() {
        let config = model("[model]\nneighborhood_size_k=42\n");
        assert_eq!(config.knn.algorithm, Algorithm::Vmis);
        assert_eq!(config.neighborhood_size_k, 42);
        assert_eq!(config.idf_weighting, 1.0);
    }
    #[test]
    fn selected_model_uses_reference_defaults_and_explicit_overrides() {
        let config = model("[model]\nalgorithm=\"vstan\"\n[model.knn]\nlambda_snh=none\nlambda_idf=2.5\nremind=false\n");
        assert_eq!(config.knn.algorithm, Algorithm::VsknnStan);
        assert_eq!(config.knn.lambda_snh, None);
        assert_eq!(config.knn.lambda_spw, Some(1.02));
        assert_eq!(config.knn.lambda_idf, Some(2.5));
        assert!(!config.knn.remind);
    }
    #[test]
    #[should_panic(expected = "Invalid model.algorithm")]
    fn unavailable_user_aware_algorithm_is_rejected() {
        model("[model]\nalgorithm=ustan\n");
    }
    #[test]
    #[should_panic(expected = "Invalid kNN scoring configuration")]
    fn invalid_decay_is_rejected() {
        model("[model]\nalgorithm=stan\n[model.knn]\nlambda_inh=0\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use justconfig::sources::defaults::Defaults;

    fn server_value(key: &str, value: &str) -> anyhow::Result<ServerConfig> {
        let mut config = Config::default();
        let mut defaults = Defaults::default();
        defaults.set(ConfPath::from(&["server", key]), value, "test");
        config.add_source(defaults);
        ServerConfig::parse(&config, ConfPath::from(&["server"]))
    }

    #[test]
    fn supplied_invalid_numbers_do_not_use_defaults() {
        for key in ["port", "num_workers"] {
            let error = server_value(key, "invalid").err().unwrap();
            assert!(error.to_string().contains(key));
        }
        assert_eq!(server_value("host", "localhost").unwrap().port, 8080);
    }

    #[test]
    fn explicit_missing_configuration_is_an_error() {
        assert!(AppConfig::try_new("missing-configuration-test.ini".into())
            .err()
            .unwrap()
            .to_string()
            .contains("open configuration"));
    }
}
