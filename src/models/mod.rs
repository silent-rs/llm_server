use crate::configs::Config;
use crate::models::audio::whisper::Whisper;
use crate::models::chat::ChatModel;
use std::collections::HashMap;

pub(crate) mod audio;
pub(crate) mod chat;
mod device;

#[derive(Debug, Clone)]
pub struct Models {
    model_map: HashMap<String, Model>,
}

impl Models {
    pub fn new(config: Config) -> anyhow::Result<Self> {
        let mut model_map = HashMap::new();
        for (index, chat_config) in config
            .chat_configs
            .unwrap_or_default()
            .into_iter()
            .enumerate()
        {
            let start = std::time::Instant::now();
            println!("{}: init chat model: {}", index + 1, chat_config.alias);
            let alias = chat_config.alias.clone();
            let model = chat::init_model(chat_config.clone())?;
            println!(
                "init chat model: {} finished in {:2}s",
                chat_config.alias,
                start.elapsed().as_secs()
            );
            model_map.insert(alias, Model::Chat(model));
        }
        for (index, whisper_config) in config
            .whisper_configs
            .unwrap_or_default()
            .into_iter()
            .enumerate()
        {
            let start = std::time::Instant::now();
            println!(
                "{}: init whisper model: {}",
                index + 1,
                whisper_config.alias
            );
            let alias = whisper_config.alias.clone();
            let model = audio::whisper::init_model(whisper_config.clone())?;
            println!(
                "init whisper model: {} finished in {:2}s",
                whisper_config.alias,
                start.elapsed().as_secs()
            );
            model_map.insert(alias, Model::Whisper(model));
        }
        Ok(Self { model_map })
    }
    pub(crate) fn get_whisper(&self, alias: String) -> Option<&Whisper> {
        match self.model_map.get(&alias) {
            Some(Model::Whisper(model)) => Some(model),
            _ => None,
        }
    }
    pub(crate) fn get_chat(&self, alias: String) -> Option<&ChatModel> {
        match self.model_map.get(&alias) {
            Some(Model::Chat(model)) => Some(model),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Model {
    Whisper(Whisper),
    Chat(ChatModel),
}
