pub(crate) mod decoder;
pub(crate) mod model;
mod multilingual;
mod pcm_decode;

use crate::configs::WhisperModelConfig;
use crate::models::audio::whisper::decoder::{Decoder, Task};
use crate::models::audio::whisper::pcm_decode::pcm_decode;
use crate::models::device::{device, token_id};
use crate::types::audio::transcription::{CreateTranscriptionRequest, CreateTranscriptionResponse};
use anyhow::{Error as E, Result};
use candle_core as candle;
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, audio, Config};
use model::Model;
use silent::prelude::info;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Clone, Debug)]
pub(crate) struct Whisper {
    tokenizer: Tokenizer,
    model: Model,
    config: Config,
    mel_filters: Vec<f32>,
    device: candle::Device,
    seed: u64,
}

impl Whisper {
    pub(crate) fn handle(
        &self,
        request: CreateTranscriptionRequest,
        task: Option<Task>,
    ) -> Result<CreateTranscriptionResponse> {
        let input = request.file.path().clone();

        let pcm_data = pcm_decode(input)?;
        let config = self.config.clone();
        let mel_filters = self.mel_filters.clone();
        let mel = audio::pcm_to_mel(&config, &pcm_data, &mel_filters);
        let mel_len = mel.len();
        let mel = Tensor::from_vec(
            mel,
            (
                1,
                self.config.num_mel_bins,
                mel_len / self.config.num_mel_bins,
            ),
            &self.device,
        )?;
        info!("loaded mel: {:?}", mel.dims());
        let mut model = self.model.clone();
        let tokenizer = self.tokenizer.clone();
        let device = self.device.clone();
        let language_token = match (request.model.is_multilingual(), request.language) {
            (true, None) => Some(multilingual::detect_language(&mut model, &tokenizer, &mel)?),
            (false, None) => None,
            (true, Some(language)) => match token_id(&tokenizer, &format!("<|{language}|>")) {
                Ok(token_id) => Some(token_id),
                Err(_) => anyhow::bail!("language {language} is not supported"),
            },
            (false, Some(_)) => {
                anyhow::bail!("a language cannot be set for non-multilingual models")
            }
        };
        info!("matched language: {:?}", language_token);
        let mut dc = Decoder::new(
            model,
            tokenizer,
            self.seed,
            &device,
            language_token,
            task,
            request.response_format.has_timestamps(),
            request.response_format.is_verbose(),
            request.temperature,
        )?;
        let segments = dc.run(&mel)?;
        Ok(CreateTranscriptionResponse::new(
            segments,
            request.response_format.clone(),
        ))
    }
}

pub(crate) fn init_model(args: WhisperModelConfig) -> Result<Whisper> {
    let device = device(args.cpu)?;
    let model_id = args.model_id;
    let (config_filename, tokenizer_filename, weights_filename) = {
        let config = PathBuf::from(format!("{model_id}/config.json"));
        let tokenizer = PathBuf::from(format!("{model_id}/tokenizer.json"));
        let model = PathBuf::from(format!("{model_id}/model.safetensors"));
        (config, tokenizer, model)
    };

    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let mel_bytes = match config.num_mel_bins {
        80 => include_bytes!("melfilters.bytes").as_slice(),
        128 => include_bytes!("melfilters128.bytes").as_slice(),
        nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);

    let model = {
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)? };
        Model::Normal(m::model::Whisper::load(&vb, config.clone())?)
    };
    Ok(Whisper {
        tokenizer,
        model,
        config,
        mel_filters,
        device,
        seed: args.seed,
    })
}
