use crate::configs::ChatModelConfig;

pub(crate) mod chat_format;
mod model;
mod utils;

pub(crate) use model::{init_model, ChatModel};
