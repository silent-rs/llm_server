use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub(crate) chat_configs: Vec<ChatModelConfig>,
    pub(crate) whisper_configs: Vec<WhisperModelConfig>,
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
    pub(crate) context_size: Option<i32>,
    pub(crate) seed: Option<i32>,
    pub(crate) n_batch: Option<i32>,
    pub(crate) f16_memory: Option<bool>,
    pub(crate) m_lock: Option<bool>,
    pub(crate) m_map: Option<bool>,
    pub(crate) low_vram: Option<bool>,
    pub(crate) vocab_only: Option<bool>,
    pub(crate) embeddings: Option<bool>,
    pub(crate) n_gpu_layers: Option<i32>,
    pub(crate) main_gpu: Option<String>,
    pub(crate) tensor_split: Option<String>,
    pub(crate) numa: Option<bool>,
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
}
