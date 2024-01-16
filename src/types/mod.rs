use crate::types::audio::transcription::{CreateTranscriptionRequest, CreateTranscriptionResponse};
use tokio::sync::oneshot::Sender;

pub(crate) mod audio;
mod chat;

#[derive(Debug)]
pub enum RequestTypes {
    Whisper(CreateTranscriptionRequest, Sender<ResponseTypes>),
    Chat,
}

impl From<(CreateTranscriptionRequest, Sender<ResponseTypes>)> for RequestTypes {
    fn from(value: (CreateTranscriptionRequest, Sender<ResponseTypes>)) -> Self {
        Self::Whisper(value.0, value.1)
    }
}

#[derive(Debug, Clone)]
pub enum ResponseTypes {
    Whisper(CreateTranscriptionResponse),
    Chat,
}

impl From<CreateTranscriptionResponse> for ResponseTypes {
    fn from(response: CreateTranscriptionResponse) -> Self {
        Self::Whisper(response)
    }
}
