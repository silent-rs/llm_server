use crate::types::chat::completion::{ChatCompleteUsage, ChatCompletionChoice};
use chrono::Local;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionResponse {
    /// A unique identifier for the chat completion.
    pub id: String,
    /// A list of chat completion choices. Can be more than one if n is greater than 1.
    pub choices: Vec<ChatCompletionChoice>,
    /// The Unix timestamp (in seconds) of when the chat completion was created.
    pub created: usize,
    /// The model used for the chat completion.
    pub model: String,
    /// This fingerprint represents the backend configuration that the model runs with. Can be used in conjunction with the seed request parameter to understand when backend changes have been made that might impact determinism.
    pub system_fingerprint: String,
    /// The object type, which is always chat.completion.
    pub object: String,
    /// Usage statistics for the completion request.
    pub usage: ChatCompleteUsage,
}

impl ChatCompletionResponse {
    pub(crate) fn new(model: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            choices: Vec::new(),
            created: Local::now().timestamp() as usize,
            model,
            system_fingerprint: String::new(),
            object: "chat.completion".to_string(),
            usage: ChatCompleteUsage {
                completion_tokens: 0,
                prompt_tokens: 0,
                total_tokens: 0,
            },
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionResponseChunk {
    /// A unique identifier for the chat completion.
    pub id: String,
    /// A list of chat completion choices. Can be more than one if n is greater than 1.
    pub choices: Vec<ChatCompletionChoice>,
    /// The Unix timestamp (in seconds) of when the chat completion was created.
    pub created: usize,
    /// The model used for the chat completion.
    pub model: String,
    /// This fingerprint represents the backend configuration that the model runs with. Can be used in conjunction with the seed request parameter to understand when backend changes have been made that might impact determinism.
    pub system_fingerprint: String,
    /// The object type, which is always chat.completion.
    pub object: String,
}

impl ChatCompletionResponseChunk {
    pub(crate) fn from_response(
        response: &ChatCompletionResponse,
        choices: Vec<ChatCompletionChoice>,
    ) -> Self {
        let ChatCompletionResponse {
            id,
            created,
            model,
            system_fingerprint,
            ..
        } = response.clone();
        Self {
            id,
            choices,
            created,
            model,
            system_fingerprint,
            object: "chat.completion.chunk".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum ChatResponse {
    Chunk(ChatCompletionResponseChunk),
    Completion(ChatCompletionResponse),
    Text(String),
}

impl From<ChatCompletionResponse> for ChatResponse {
    fn from(value: ChatCompletionResponse) -> Self {
        Self::Completion(value)
    }
}

impl From<ChatCompletionResponseChunk> for ChatResponse {
    fn from(value: ChatCompletionResponseChunk) -> Self {
        Self::Chunk(value)
    }
}
