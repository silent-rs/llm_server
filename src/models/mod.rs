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
        for chat_config in config.chat_configs.unwrap_or_default() {
            let alias = chat_config.alias.clone();
            let model = chat::init_model(chat_config)?;
            model_map.insert(alias, Model::Chat(model));
        }
        for whisper_config in config.whisper_configs.unwrap_or_default() {
            let alias = whisper_config.alias.clone();
            let model = audio::whisper::init_model(whisper_config)?;
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
