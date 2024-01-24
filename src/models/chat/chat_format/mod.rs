mod alpaca;
mod chatglm3;
mod chatml;
mod llama2;
mod openchat;

use crate::types::chat::completion::{
    AssistantMessage, ChatCompletionMessage, SystemMessage, ToolMessage, UserMessage,
};
use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub(crate) enum ChatFormat {
    #[serde(rename = "llama-2")]
    Llama2,
    #[serde(rename = "alpaca")]
    Alpaca,
    #[serde(rename = "chatml")]
    ChatML,
    #[serde(rename = "chatglm3")]
    ChatGLM3,
    #[serde(rename = "openchat")]
    OpenChat,
}

pub enum ChatMessage {
    /// A message from a human.
    User(UserMessage),
    /// A message from the assistant.
    Assistant(AssistantMessage),
    /// A message from a tool.
    Tool(ToolMessage),
}
struct ChatMessages {
    system: SystemMessage,
    chat: Vec<ChatMessage>,
}

impl TryFrom<Vec<ChatCompletionMessage>> for ChatMessages {
    type Error = anyhow::Error;

    fn try_from(value: Vec<ChatCompletionMessage>) -> Result<Self, Self::Error> {
        let mut system = None;
        let mut chat = vec![];
        for message in value {
            match message {
                ChatCompletionMessage::System(message) => {
                    system = Some(message);
                }
                ChatCompletionMessage::User(message) => {
                    chat.push(ChatMessage::User(message));
                }
                ChatCompletionMessage::Assistant(message) => {
                    chat.push(ChatMessage::Assistant(message));
                }
                ChatCompletionMessage::Tool(message) => {
                    chat.push(ChatMessage::Tool(message));
                }
            }
        }
        Ok(Self {
            system: system.ok_or_else(|| anyhow::anyhow!("No system message found"))?,
            chat,
        })
    }
}

impl ChatFormat {
    pub(crate) fn format_messages(&self, messages: Vec<ChatCompletionMessage>) -> Result<String> {
        let messages = messages.try_into()?;
        match self {
            Self::Llama2 => llama2::format_messages(messages),
            Self::Alpaca => alpaca::format_messages(messages),
            Self::ChatML => chatml::format_messages(messages),
            Self::ChatGLM3 => chatglm3::format_messages(messages),
            Self::OpenChat => openchat::format_messages(messages),
        }
    }
    pub(crate) fn get_eos_token(&self) -> String {
        match self {
            Self::Llama2 => "</s>".to_string(),
            Self::Alpaca => "</s>".to_string(),
            // Self::ChatML => "<|im_end|>".to_string(),
            Self::ChatML => "<".to_string(),
            Self::ChatGLM3 => "</s>".to_string(),
            Self::OpenChat => "<|end_of_turn|>".to_string(),
        }
    }
}
