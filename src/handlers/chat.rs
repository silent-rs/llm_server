use crate::models::Model;
use crate::types::chat::completion::ChatCompletionRequest;
use crate::types::{RequestTypes, ResponseTypes};
use silent::prelude::info;
use silent::{Request, Response, SilentError, StatusCode};
use std::sync::Arc;
use tokio::sync::mpsc::Sender;

pub(crate) async fn chat_completions(mut req: Request) -> silent::Result<Response> {
    let sender = req.configs().get::<Sender<RequestTypes>>().unwrap().clone();
    let req: ChatCompletionRequest = req.json_parse().await?;

    let (tx, mut rx) = tokio::sync::mpsc::channel::<ResponseTypes>(20);
    sender.send((req, tx).into()).await.map_err(|e| {
        SilentError::business_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to send request: {}", e),
        )
    })?;
    info!("sent request");
    match rx.recv().await {
        Some(ResponseTypes::Chat(res)) => Ok(res.into()),
        // Err(e) => {
        //     info!("error receiving response: {:?}", e);
        //     Err(SilentError::business_error(
        //         StatusCode::INTERNAL_SERVER_ERROR,
        //         format!("failed to create transcription: {e}"),
        //     ))
        // }
        _ => Err(SilentError::business_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "failed to create transcription".to_string(),
        )),
    }
}
