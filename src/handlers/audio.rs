use crate::types::audio::transcription::CreateTranscriptionRequest;
use crate::types::{RequestTypes, ResponseTypes};
use silent::{Request, Response, Result, SilentError, StatusCode};
use tokio::sync::mpsc::Sender;

pub(crate) async fn create_transcription(mut req: Request) -> Result<Response> {
    let sender = req.configs().get::<Sender<RequestTypes>>().unwrap().clone();
    let req: CreateTranscriptionRequest = req.form_data().await?.try_into().map_err(|e| {
        SilentError::business_error(
            StatusCode::BAD_REQUEST,
            format!("failed to parse request: {}", e),
        )
    })?;
    let (tx, rx) = tokio::sync::oneshot::channel::<ResponseTypes>();
    sender.send((req, tx).into()).await.map_err(|e| {
        SilentError::business_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to send request: {}", e),
        )
    })?;
    if let Ok(ResponseTypes::Whisper(res)) = rx.await {
        Ok(res.into())
    } else {
        Err(SilentError::business_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "failed to create transcription".to_string(),
        ))
    }
}
