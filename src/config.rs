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

pub struct AppConfig {
    pub server: ServerConfig,
    pub log: LogConfig,
    pub data: DataConfig,
    pub model: ModelConfig,
    pub logic: LogicConfig,
    pub hyperparam: HyperparamConfig
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
    pub m_most_recent_sessions: usize,
    pub neighborhood_size_k: usize,
    pub num_items_to_recommend: usize,
    pub max_items_in_session: usize,
}

pub struct LogicConfig {
    pub enable_business_logic: bool,
}

pub struct HyperparamConfig {
    pub training_data_path: String,
    pub test_data_path: String,
    pub validation_data_path: String,
    pub num_iterations: usize,
    pub save_records: bool,
    pub out_path: String,
    pub enable_business_logic: bool,
    pub n_most_recent_sessions_choices: String,
    pub neighborhood_size_k_choices: String,
    pub last_items_in_session_choices: String
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
            hyperparam: HyperparamConfig::parse(&conf, ConfPath::from(&["hyperparam"]))
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
        }
    }
}

impl LogicConfig {
    fn parse(conf: &Config, path: ConfPath) -> LogicConfig {
        LogicConfig {
            enable_business_logic: conf
                .get(path.push("enable_business_logic"))
                .unquote()
                .value()
                .unwrap(),
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
            save_records: conf
                .get(path.push("save_records"))
                .trim()
                .value()
                .unwrap(),
            out_path: conf
                .get(path.push("out_path"))
                .unquote()
                .value()
                .unwrap(),
            enable_business_logic: conf
                .get(path.push("enable_business_logic"))
                .trim()
                .value()
                .unwrap(),
            n_most_recent_sessions_choices: conf
                .get(path.push("n_most_recent_sessions_choices"))
                .trim()
                .value()
                .unwrap(),
            neighborhood_size_k_choices: conf
                .get(path.push("neighborhood_size_k_choices"))
                .trim()
                .value()
                .unwrap(),
            last_items_in_session_choices: conf
                .get(path.push("last_items_in_session_choices"))
                .trim()
                .value()
                .unwrap(),
        }
    }
}
