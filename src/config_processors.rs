use justconfig::error::ConfigError;
use justconfig::item::{MapAction, StringItem};

/// Remove quotes from configuration strings.
pub trait Unquote
where
    Self: Sized,
{
    fn unquote(self) -> Result<StringItem, ConfigError>;
}

impl Unquote for Result<StringItem, ConfigError> {
    /// Call this method to remove quotes around all configuration values.
    ///
    /// All configuration values will automatically be trimmed and checked for a
    /// loading and trailing quote (`"`). If the quote is there, it will be
    /// removed. If it's missing a `ProcessingError::MissingQuotes` error will be
    /// generated.
    ///
    /// ## Example
    ///
    /// ```rust
    /// # use justconfig::Config;
    /// # use justconfig::ConfPath;
    /// # use justconfig::item::ValueExtractor;
    /// # use justconfig::sources::defaults::Defaults;
    /// # use justconfig::processors::Unquote;
    /// #
    /// # let mut conf = Config::default();
    /// # let mut defaults = Defaults::default();
    /// defaults.set(conf.root().push_all(&["quoted"]), "\"abc\"", "source info");
    /// conf.add_source(defaults);
    ///
    /// let value: String = conf.get(ConfPath::from(&["quoted"])).unquote().value().unwrap();
    ///
    /// assert_eq!(value, "abc");
    /// ```
    fn unquote(self) -> Result<StringItem, ConfigError> {
        self?.map(|v| {
            let v = v.trim();

            if v.starts_with('"') && v.ends_with('"') {
                MapAction::Replace(vec![v[1..v.len() - 1].to_owned()])
            } else {
                MapAction::Keep
            }
        })
    }
}
