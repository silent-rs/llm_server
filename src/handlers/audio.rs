use crate::types::audio::transcription::CreateTranscriptionRequest;
use crate::Models;
use silent::{Request, Response, Result, SilentError, StatusCode};

pub(crate) async fn create_transcription(mut req: Request) -> Result<Response> {
    let transcription_req: CreateTranscriptionRequest =
        req.form_data().await?.try_into().map_err(|e| {
            SilentError::business_error(
                StatusCode::BAD_REQUEST,
                format!("failed to parse request: {}", e),
            )
        })?;
    let model = req.get_config::<Models>()?;
    let whisper_model = model
        .get_whisper(transcription_req.model.get_model_string())
        .ok_or_else(|| {
            SilentError::business_error(StatusCode::BAD_REQUEST, "model not set".to_string())
        })?;
    let result = whisper_model.handle(transcription_req, None).map_err(|e| {
        SilentError::business_error(
            StatusCode::BAD_REQUEST,
            format!("failed to handle whisper model: {}", e),
        )
    })?;
    Ok(result.into())
}
