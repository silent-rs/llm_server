use crate::configs::Config;
use crate::models::audio::whisper::Whisper;
use crate::types::RequestTypes;
use llama_cpp_rs::LLama;
use silent::prelude::{error, info};
use std::collections::HashMap;
use tokio::signal;
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
        loop {
            #[cfg(unix)]
            let terminate = async {
                signal::unix::signal(signal::unix::SignalKind::terminate())
                    .expect("failed to install signal handler")
                    .recv()
                    .await;
            };

            #[cfg(not(unix))]
            let terminate = async {
                let _ = std::future::pending::<()>().await;
            };
            tokio::select! {
                    _ = signal::ctrl_c() => {
                        info!("received ctrl-c");
                        break Ok(());
                    }
                    _ = terminate => {
                        info!("received terminate");
                        break Ok(());
                    }
                    Some(types) = self.receiver.recv() => {
                info!("received request");
                match types {
                    RequestTypes::Whisper(request, sender) => {
                        let model = self.get_whisper(request.model.get_model_string());
                        if let Some(model) = model {
                            info!("model found");
                            let res = model.handle(request, None)?;
                            info!("response created");
                            match sender.send(res.into()) {
                                Ok(_) => info!("response sent"),
                                Err(e) => error!("error sending response: {:?}", e),
                            }
                        }
                    }
                    RequestTypes::Chat => {}
                }
            }}
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Model {
    Whisper(Whisper),
    Chat(LLama),
}
