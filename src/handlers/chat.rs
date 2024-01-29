use crate::types::chat::response::ChatResponse;
use crate::types::chat::{
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChunk,
};
use crate::Models;
use silent::prelude::{sse_reply, warn, SSEEvent};
use silent::{Request, Response, SilentError, StatusCode};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

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

    // if chat_completion_req.stream.clone().unwrap_or(false) {
    //     let (tx, rx) = mpsc::unbounded_channel();
    //     let _ = chat_model
    //         .stream_handle(chat_completion_req, tx.clone())
    //         .map_err(|e| {
    //             SilentError::business_error(
    //                 StatusCode::BAD_REQUEST,
    //                 format!("failed to handle chat model: {}", e),
    //             )
    //         })?;
    //     let rx = UnboundedReceiverStream::new(rx);
    //     let stream = rx.map(|msg| match msg {
    //         ChatResponse::Chunk(chunk) => {
    //             Ok(SSEEvent::default().data(serde_json::to_string(&chunk)?))
    //         }
    //         ChatResponse::Completion(completion) => {
    //             Ok(SSEEvent::default().data(serde_json::to_string(&completion)?))
    //         }
    //         ChatResponse::Text(text) => Ok(SSEEvent::default().data(text)),
    //     });
    //     let result = sse_reply(stream);
    //     Ok(result)
    // } else {
    //     let result = chat_model.handle(chat_completion_req).map_err(|e| {
    //         SilentError::business_error(
    //             StatusCode::BAD_REQUEST,
    //             format!("failed to handle chat model: {}", e),
    //         )
    //     })?;
    //     Ok(result.into())
    // }
}
