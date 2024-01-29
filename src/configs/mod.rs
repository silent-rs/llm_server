use crate::models::chat::chat_format::ChatFormat;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub(crate) chat_configs: Option<Vec<ChatModelConfig>>,
    pub(crate) whisper_configs: Option<Vec<WhisperModelConfig>>,
}

impl Config {
    pub fn load(path: String) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config = toml::from_str(&contents)?;
        Ok(config)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ChatModelConfig {
    pub(crate) model_id: String,
    pub(crate) alias: String,
    pub(crate) tokenizer: Option<String>,
    #[serde(default = "default_cpu")]
    pub(crate) cpu: bool,
    #[serde(default = "default_seed")]
    pub(crate) seed: u64,
    pub(crate) gqa: usize,
    #[serde(default = "default_chat_format")]
    pub(crate) chat_format: ChatFormat,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct WhisperModelConfig {
    pub(crate) model_id: String,
    pub(crate) alias: String,
    #[serde(default = "default_cpu")]
    pub(crate) cpu: bool,
    #[serde(default = "default_seed")]
    pub(crate) seed: u64,
    #[serde(default = "default_quantized")]
    pub(crate) quantized: bool,
}

fn default_chat_format() -> ChatFormat {
    ChatFormat::ChatML
}
fn default_cpu() -> bool {
    true
}

fn default_seed() -> u64 {
    299792458
}

fn default_quantized() -> bool {
    true
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_config() {
        let config = super::Config::load("test_config.toml".to_string()).unwrap();
        println!("{:?}", config);
    }
    #[test]
    fn test_chat_config() {
        let config = super::Config::load("test_chat_config.toml".to_string()).unwrap();
        println!("{:?}", config);
    }
}
