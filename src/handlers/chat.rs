use crate::types::chat::completion::ChatCompletionRequest;
use crate::Models;
use silent::{Request, Response, SilentError, StatusCode};

pub(crate) async fn chat_completions(mut req: Request) -> silent::Result<Response> {
    let chat_completion_req: ChatCompletionRequest = req.json_parse().await?;
    let model = req.get_config::<Models>()?;

    let chat_model = model
        .get_chat(chat_completion_req.model.clone())
        .ok_or_else(|| {
            SilentError::business_error(StatusCode::BAD_REQUEST, "model not set".to_string())
        })?;
    let result = chat_model.handle(chat_completion_req).map_err(|e| {
        SilentError::business_error(
            StatusCode::BAD_REQUEST,
            format!("failed to handle chat model: {}", e),
        )
    })?;
    Ok(result.into())
}
