use crate::types::audio::transcription::{CreateTranscriptionRequest, CreateTranscriptionResponse};
use crate::types::chat::completion::{ChatCompletionRequest, ChatCompletionResponse};
use tokio::sync::mpsc::Sender;

pub(crate) mod audio;
pub(crate) mod chat;

#[derive(Debug, Clone)]
pub enum RequestTypes {
    Whisper(CreateTranscriptionRequest, Sender<ResponseTypes>),
    Chat(ChatCompletionRequest, Sender<ResponseTypes>),
}

impl From<(CreateTranscriptionRequest, Sender<ResponseTypes>)> for RequestTypes {
    fn from(value: (CreateTranscriptionRequest, Sender<ResponseTypes>)) -> Self {
        Self::Whisper(value.0, value.1)
    }
}

impl From<(ChatCompletionRequest, Sender<ResponseTypes>)> for RequestTypes {
    fn from(value: (ChatCompletionRequest, Sender<ResponseTypes>)) -> Self {
        Self::Chat(value.0, value.1)
    }
}

#[derive(Debug, Clone)]
pub enum ResponseTypes {
    Whisper(CreateTranscriptionResponse),
    Chat(ChatCompletionResponse),
}

impl From<CreateTranscriptionResponse> for ResponseTypes {
    fn from(response: CreateTranscriptionResponse) -> Self {
        Self::Whisper(response)
    }
}

impl From<ChatCompletionResponse> for ResponseTypes {
    fn from(response: ChatCompletionResponse) -> Self {
        Self::Chat(response)
    }
}
