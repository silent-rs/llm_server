use crate::types::chat::completion::{
    ChatCompletionChoice, ChatResponseFormat, ChatResponseFormatObject,
};
use crate::types::chat::response::ChatResponse;
use crate::types::chat::ChatCompletionRequest;
use crate::Models;
use silent::prelude::sse_reply;
use silent::{Request, Response, SilentError, StatusCode};

pub(crate) async fn chat_completions(mut req: Request) -> silent::Result<Response> {
    let chat_completion_req: ChatCompletionRequest = req.json_parse().await?;
    let model = req.get_config::<Models>()?;

    let chat_model = model
        .get_chat(chat_completion_req.model.clone())
        .ok_or_else(|| {
            SilentError::business_error(StatusCode::BAD_REQUEST, "model not set".to_string())
        })?;

    if chat_completion_req.stream.clone().unwrap_or(false) {
        let stream = chat_model.stream_handle(chat_completion_req).map_err(|e| {
            SilentError::business_error(
                StatusCode::BAD_REQUEST,
                format!("failed to handle chat model: {}", e),
            )
        })?;
        let result = sse_reply(stream);
        Ok(result)
    } else {
        let result = chat_model
            .handle(chat_completion_req.clone())
            .map_err(|e| {
                SilentError::business_error(
                    StatusCode::BAD_REQUEST,
                    format!("failed to handle chat model: {}", e),
                )
            })?;
        match chat_completion_req.response_format {
            None => Ok(result.into()),
            Some(format) => {
                if format.r#type == ChatResponseFormat::Json {
                    Ok(result.into())
                } else {
                    let result = match result.choices.first() {
                        None => "".to_string(),
                        Some(choice) => choice.message.content.clone().unwrap_or("".to_string()),
                    };
                    Ok(result.into())
                }
            }
        }
    }
}
