use super::{ChatMessage, ChatMessages};
use anyhow::Result;

pub(crate) fn format_messages(messages: ChatMessages) -> Result<String> {
    let system_message = format!("{}\n\n", messages.system.content);
    let message = messages
        .chat
        .into_iter()
        .map(transform)
        .collect::<Vec<String>>()
        .join("</s>");
    Ok(format!("{}\n{}\n</s>", system_message, message))
}

fn transform(message: ChatMessage) -> String {
    match message {
        ChatMessage::User(message) => format!("### Instruction:\n{}\n", message.content),
        ChatMessage::Assistant(message) => {
            format!(
                "### Response\n{}\n",
                message.content.unwrap_or("".to_string())
            )
        }
        // todo: implement tool
        ChatMessage::Tool(_) => !unimplemented!("tool"),
    }
}
