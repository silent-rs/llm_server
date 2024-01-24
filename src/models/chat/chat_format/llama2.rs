use super::{ChatMessage, ChatMessages};
use anyhow::Result;

pub(crate) fn format_messages(messages: ChatMessages) -> Result<String> {
    let system_message = format!("<s>[INST] <<SYS>>\n{}\n<</SYS>>", messages.system.content);
    let message = messages
        .chat
        .into_iter()
        .map(transform)
        .collect::<Vec<String>>()
        .join("")
        .as_str()
        .replacen("<s>[INST]\n", "", 1);
    Ok(format!("{}\n{}", system_message, message))
}

fn transform(message: ChatMessage) -> String {
    match message {
        ChatMessage::User(message) => format!("<s>[INST]\n{}\n[/INST]", message.content),
        ChatMessage::Assistant(message) => {
            format!(
                "\n{}",
                match message.content {
                    None => {
                        "".to_string()
                    }
                    Some(content) => {
                        match content.as_str() {
                            "" => "".to_string(),
                            _ => {
                                format!("{}\n</s>\n", content)
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
            r#"<s>[INST] <<SYS>>
you are helpfull assistant!
<</SYS>>
Hello
[/INST]
World
</s>
<s>[INST]
who are you
[/INST]"#
        )
    }
}
