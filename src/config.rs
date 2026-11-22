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
const DEFAULT_IDF_WEIGHTING: usize = 1;

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

pub struct ModelConfig {
    pub knn: crate::vmisknn::algorithms::KnnConfig,
    pub m_most_recent_sessions: usize,
    pub neighborhood_size_k: usize,
    pub num_items_to_recommend: usize,
    pub max_items_in_session: usize,
    pub idf_weighting: usize,
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
    pub save_records: bool,
    pub out_path: String,
    pub enable_business_logic: bool,
    pub n_most_recent_sessions_range: String,
    pub neighborhood_size_k_range: String,
    pub last_items_in_session_range: String,
    pub idf_weighting_range: String,
}

impl AppConfig {
    pub fn new(config_path: String) -> AppConfig {
        // Initialize config object
        let mut conf = Config::default();

        // Check if there is a config file
        if let Ok(config_file) = File::open(&config_path) {
            let config_text = ConfigText::new(config_file, &config_path)
                .expect("Loading configuration file failed.");
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
        AppConfig::parse(conf)
    }

    fn parse(conf: justconfig::Config) -> AppConfig {
        AppConfig {
            server: ServerConfig::parse(&conf, ConfPath::from(&["server"])),
            log: LogConfig::parse(&conf, ConfPath::from(&["log"])),
            data: DataConfig::parse(&conf, ConfPath::from(&["data"])),
            model: ModelConfig::parse(&conf, ConfPath::from(&["model"])),
            logic: LogicConfig::parse(&conf, ConfPath::from(&["logic"])),
            hyperparam: HyperparamConfig::parse(&conf, ConfPath::from(&["hyperparam"])),
        }
    }
}

impl ServerConfig {
    fn parse(conf: &Config, path: ConfPath) -> ServerConfig {
        ServerConfig {
            host: conf
                .get(path.push("host"))
                .unquote()
                .value()
                .unwrap_or_else(|_| String::from("0.0.0.0")),
            port: conf.get(path.push("port")).trim().value().unwrap_or(8080),
            num_workers: conf
                .get(path.push("num_workers"))
                .trim()
                .value()
                // Detect number of CPUs
                .unwrap_or_else(|_| sys_info::cpu_num().unwrap_or_default().try_into().unwrap()),
        }
    }
}

impl LogConfig {
    fn parse(conf: &Config, path: ConfPath) -> LogConfig {
        LogConfig {
            level: conf
                .get(path.push("level"))
                .unquote()
                .value()
                .unwrap_or_default(),
        }
    }
}

impl DataConfig {
    fn parse(conf: &Config, path: ConfPath) -> DataConfig {
        DataConfig {
            training_data_path: conf
                .get(path.push("training_data_path"))
                .unquote()
                .value()
                .unwrap(),
        }
    }
}

impl ModelConfig {
    fn parse(conf: &Config, path: ConfPath) -> ModelConfig {
        ModelConfig {
            knn: parse_knn_config(conf, &path),
            m_most_recent_sessions: conf
                .get(path.push("m_most_recent_sessions"))
                .trim()
                .value()
                .unwrap_or(DEFAULT_MOST_RECENT_SESSIONS_M),
            neighborhood_size_k: conf
                .get(path.push("neighborhood_size_k"))
                .trim()
                .value()
                .unwrap_or(DEFAULT_NEIGHBORHOOD_SIZE_K),
            num_items_to_recommend: conf
                .get(path.push("num_items_to_recommend"))
                .trim()
                .value()
                .unwrap_or(DEFAULT_NUM_ITEMS_TO_RECOMMEND),
            max_items_in_session: conf
                .get(path.push("max_items_in_session"))
                .trim()
                .value()
                .unwrap_or(DEFAULT_MAX_ITEMS_IN_SESSION),
            idf_weighting: conf
                .get(path.push("idf_weighting"))
                .trim()
                .value()
                .unwrap_or(DEFAULT_IDF_WEIGHTING),
        }
    }
}

impl LogicConfig {
    fn parse(conf: &Config, path: ConfPath) -> LogicConfig {
        LogicConfig {
            enable_business_logic: conf
                .get(path.push("enable_business_logic"))
                .trim()
                .value()
                .unwrap(),
            example_item_id: conf
                .get(path.push("example_item_id"))
                .trim()
                .value()
                .unwrap_or(0),
        }
    }
}

impl HyperparamConfig {
    fn parse(conf: &Config, path: ConfPath) -> HyperparamConfig {
        HyperparamConfig {
            training_data_path: conf
                .get(path.push("training_data_path"))
                .unquote()
                .value()
                .unwrap(),
            test_data_path: conf
                .get(path.push("test_data_path"))
                .unquote()
                .value()
                .unwrap(),
            validation_data_path: conf
                .get(path.push("validation_data_path"))
                .unquote()
                .value()
                .unwrap(),
            num_iterations: conf
                .get(path.push("num_iterations"))
                .trim()
                .value()
                .unwrap(),
            save_records: conf.get(path.push("save_records")).trim().value().unwrap(),
            out_path: conf.get(path.push("out_path")).unquote().value().unwrap(),
            enable_business_logic: conf
                .get(path.push("enable_business_logic"))
                .trim()
                .value()
                .unwrap(),
            n_most_recent_sessions_range: conf
                .get(path.push("n_most_recent_sessions_range"))
                .trim()
                .value()
                .unwrap(),
            neighborhood_size_k_range: conf
                .get(path.push("neighborhood_size_k_range"))
                .trim()
                .value()
                .unwrap(),
            last_items_in_session_range: conf
                .get(path.push("last_items_in_session_range"))
                .trim()
                .value()
                .unwrap(),
            idf_weighting_range: conf
                .get(path.push("idf_weighting_range"))
                .trim()
                .value()
                .unwrap(),
        }
    }
}

fn knn_value(conf: &Config, path: &ConfPath, key: &str) -> Option<String> {
    let value: Result<String, _> = conf.get(path.push(key)).unquote().trim().value();
    match value {
        Ok(value) => Some(value),
        Err(justconfig::error::ConfigError::ValueNotFound(_)) => None,
        Err(error) => panic!("Invalid kNN configuration: {}", error),
    }
}
fn parse_knn_config(conf: &Config, path: &ConfPath) -> crate::vmisknn::algorithms::KnnConfig {
    use crate::vmisknn::algorithms::{Algorithm, KnnConfig};
    let algorithm = knn_value(conf, path, "algorithm")
        .map(|s| s.parse::<Algorithm>().expect("Invalid model.algorithm"))
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
        assert!(
            knn_value(conf, &options, key).is_none(),
            "model.knn.{} is not supported; see docs/knn-compatibility.md",
            key
        );
    }
    macro_rules! option {
        ($field:ident) => {
            if let Some(value) = knn_value(conf, &options, stringify!($field)) {
                config.$field = value
                    .parse()
                    .unwrap_or_else(|_| panic!("Invalid model.knn.{}", stringify!($field)));
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
            if let Some(value) = knn_value(conf, &options, stringify!($field)) {
                config.$field = if value == "none" {
                    None
                } else {
                    Some(
                        value
                            .parse()
                            .unwrap_or_else(|_| panic!("Invalid model.knn.{}", stringify!($field))),
                    )
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
        .expect("Invalid kNN scoring configuration");
    config
}

#[cfg(test)]
mod knn_config_tests {
    use super::*;
    use crate::vmisknn::algorithms::Algorithm;
    fn model(text: &str) -> ModelConfig {
        let mut config = Config::default();
        config.add_source(ConfigText::new(std::io::Cursor::new(text.as_bytes()), "test").unwrap());
        ModelConfig::parse(&config, ConfPath::from(&["model"]))
    }
    #[test]
    fn old_configuration_defaults_to_legacy_vmis() {
        let config = model("[model]\nneighborhood_size_k=42\n");
        assert_eq!(config.knn.algorithm, Algorithm::Vmis);
        assert_eq!(config.neighborhood_size_k, 42);
        assert_eq!(config.idf_weighting, 1);
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
