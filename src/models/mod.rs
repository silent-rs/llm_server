use crate::configs::Config;
use crate::models::audio::whisper::Whisper;
use crate::types::RequestTypes;
use llama_cpp_rs::LLama;
use std::collections::HashMap;
use tokio::sync::mpsc::Receiver;

pub(crate) mod audio;
mod chat;

#[derive(Debug)]
pub struct Models {
    model_map: HashMap<String, Model>,
    receiver: Receiver<RequestTypes>,
}

impl Models {
    pub fn new(config: Config, receiver: Receiver<RequestTypes>) -> anyhow::Result<Self> {
        let mut model_map = HashMap::new();
        // for chat_config in config.chat_configs {
        //     let alias = chat_config.alias.clone();
        //     let model = chat::init_model(chat_config)?;
        //     model_map.insert(chat_config.alias, Model::Chat(model));
        // }
        for whisper_config in config.whisper_configs {
            let alias = whisper_config.alias.clone();
            let model = audio::whisper::init_model(whisper_config)?;
            model_map.insert(alias, Model::Whisper(model));
        }
        Ok(Self {
            model_map,
            receiver,
        })
    }
    pub(crate) fn get_whisper(&self, alias: String) -> Option<&Whisper> {
        match self.model_map.get(&alias) {
            Some(Model::Whisper(model)) => Some(model),
            _ => None,
        }
    }

    pub async fn handle(&mut self) -> anyhow::Result<()> {
        while let Some(types) = self.receiver.recv().await {
            match types {
                RequestTypes::Whisper(request, sender) => {
                    let model = self.get_whisper(request.model.get_model_string());
                    if let Some(model) = model {
                        let res = model.handle(request, None)?;
                        sender.send(res.into()).expect("failed to send response");
                    }
                }
                RequestTypes::Chat => {}
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Model {
    Whisper(Whisper),
    Chat(LLama),
}
