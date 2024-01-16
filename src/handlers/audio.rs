use crate::types::audio::transcription::CreateTranscriptionRequest;
use crate::types::{RequestTypes, ResponseTypes};
use silent::prelude::info;
use silent::{Request, Response, Result, SilentError, StatusCode};
use tokio::sync::mpsc::Sender;
use tokio::sync::oneshot::error::RecvError;

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
    info!("sent request");
    match rx.await {
        Ok(ResponseTypes::Whisper(res)) => Ok(res.into()),
        Err(e) => {
            info!("error receiving response: {:?}", e);
            Err(SilentError::business_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to create transcription: {e}"),
            ))
        }
        _ => Err(SilentError::business_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "failed to create transcription".to_string(),
        )),
    }
}
