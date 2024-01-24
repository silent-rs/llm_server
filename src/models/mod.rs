use crate::configs::Config;
use crate::models::audio::whisper::Whisper;
use crate::models::chat::ChatModel;
use crate::types::chat::completion::{
    AssistantMessage, ChatCompleteUsage, ChatCompletionChoice, ChatCompletionResponse,
};
use crate::types::{RequestTypes, ResponseTypes};
use silent::prelude::{error, info};
use std::collections::HashMap;
use tokio::signal;
use tokio::sync::mpsc::Receiver;

pub(crate) mod audio;
pub(crate) mod chat;
mod device;

#[derive(Debug)]
pub struct Models {
    model_map: HashMap<String, Model>,
    receiver: Receiver<RequestTypes>,
}

impl Models {
    pub fn new(config: Config, receiver: Receiver<RequestTypes>) -> anyhow::Result<Self> {
        let mut model_map = HashMap::new();
        for chat_config in config.chat_configs.unwrap_or(vec![]) {
            let alias = chat_config.alias.clone();
            let model = chat::init_model(chat_config)?;
            model_map.insert(alias, Model::Chat(model));
        }
        for whisper_config in config.whisper_configs.unwrap_or(vec![]) {
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
    pub(crate) fn get_chat(&self, alias: String) -> Option<&ChatModel> {
        match self.model_map.get(&alias) {
            Some(Model::Chat(model)) => Some(model),
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
                            match sender.send(res.into()).await {
                                Ok(_) => info!("response sent"),
                                Err(e) => error!("error sending response: {:?}", e),
                            }
                        }
                    }
                    RequestTypes::Chat(request, sender) => {
                        let model = self.get_chat(request.model.clone());
                        if let Some(model) = model {
                            info!("model found");
                            let res = model.handle(request.into())?;
                            let res = ChatCompletionResponse {
                        id: "".to_string(),
                        choices: vec![ChatCompletionChoice {
                            finish_reason: Default::default(),
                            index: 0,
                            message: AssistantMessage {
                                content: Some(res),
                                name: None,
                                tool_calls: vec![],
                            },
                        }],
                        created: 0,
                        model: "".to_string(),
                        system_fingerprint: "".to_string(),
                        object: "chat.completion".to_string(),
                        usage: ChatCompleteUsage {
                            completion_tokens: 0,
                            prompt_tokens: 0,
                            total_tokens: 0,
                        },
                    };
                            info!("response created");
                            match sender.send(res.into()).await {
                                Ok(_) => info!("response sent"),
                                Err(e) => error!("error sending response: {:?}", e),
                            }
                        }
                        }
                }
            }}
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Model {
    Whisper(Whisper),
    Chat(ChatModel),
}

impl Model {
    pub(crate) fn handle(&self, request: RequestTypes) -> anyhow::Result<ResponseTypes> {
        match self {
            Model::Whisper(model) => {
                if let RequestTypes::Whisper(request, _) = request {
                    Ok(model.handle(request, None)?.into())
                } else {
                    Err(anyhow::Error::msg("invalid request type"))
                }
            }
            Model::Chat(model) => {
                if let RequestTypes::Chat(request, _) = request {
                    let result = model.handle(request.into())?;
                    let result = ChatCompletionResponse {
                        id: "".to_string(),
                        choices: vec![ChatCompletionChoice {
                            finish_reason: Default::default(),
                            index: 0,
                            message: AssistantMessage {
                                content: Some(result),
                                name: None,
                                tool_calls: vec![],
                            },
                        }],
                        created: 0,
                        model: "".to_string(),
                        system_fingerprint: "".to_string(),
                        object: "chat.completion".to_string(),
                        usage: ChatCompleteUsage {
                            completion_tokens: 0,
                            prompt_tokens: 0,
                            total_tokens: 0,
                        },
                    };
                    Ok(ResponseTypes::Chat(result).into())
                } else {
                    Err(anyhow::Error::msg("invalid request type"))
                }
            }
        }
    }
}
