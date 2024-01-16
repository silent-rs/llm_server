use crate::handlers::audio::create_transcription;
use silent::prelude::{HandlerAppend, Route};

mod audio;
mod chat;
mod model;

pub fn get_routes() -> Route {
    Route::new("").append(Route::new("/v1/audio/transcriptions").post(create_transcription))
}
