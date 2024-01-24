use super::{ChatMessage, ChatMessages};
use anyhow::Result;

pub(crate) fn format_messages(messages: ChatMessages) -> Result<String> {
    let system_message = format!(
        "<|im_start|>system\n{}\n<|im_end|>",
        messages.system.content
    );
    let message = messages
        .chat
        .into_iter()
        .map(transform)
        .collect::<Vec<String>>()
        .join("");
    Ok(format!("{}\n{}\n", system_message, message))
}

fn transform(message: ChatMessage) -> String {
    match message {
        ChatMessage::User(message) => {
            format!(
                "<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant",
                message.content
            )
        }
        ChatMessage::Assistant(message) => {
            format!(
                "\n{}\n",
                match message.content {
                    None => {
                        "".to_string()
                    }
                    Some(content) => {
                        match content.as_str() {
                            "" => "".to_string(),
                            _ => {
                                format!("{}\n<|im_end|>", content)
                            }
                        }
                    }
                }
            )
        }
        // todo: implement tool
        ChatMessage::Tool(_) => !unimplemented!("tool"),
    }
}

#[cfg(test)]
mod tests {
    use super::format_messages;
    use crate::types::chat::completion::ChatCompletionMessage;
    #[test]
    fn prompt_test() {
        let json_str = r#"
        [{
                "role": "system",
                "content": "you are helpfull assistant!"
            },
            {
        "role": "user",
                "content": "Hello"
            },{
        "role": "assistant",
                "content": "World"
            },{
        "role": "user",
                "content": "who are you"
            }
        ]"#;
        let messages: Vec<ChatCompletionMessage> = serde_json::from_str(json_str).unwrap();
        let messages = messages.try_into().unwrap();
        let prompt = format_messages(messages).unwrap();
        println!("{}", prompt);
        assert_eq!(
            prompt,
            r#"<|im_start|>system
you are helpfull assistant!
<|im_end|>
<|im_start|>user
Hello
<|im_end|>
<|im_start|>assistant
World
<|im_end|>
<|im_start|>user
who are you
<|im_end|>
<|im_start|>assistant
"#
        )
    }
}
