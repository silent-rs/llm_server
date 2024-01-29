use crate::configs::ChatModelConfig;
use crate::models::chat::chat_format::ChatFormat;
use crate::models::chat::utils::format_size;
use crate::models::device::{device, token_id};
use crate::types::chat::completion::{
    AssistantMessage, ChatCompletionChoice, ChatResponseFormat, FinishReason,
};
use crate::types::chat::{
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChunk,
};
use anyhow::{Error as E, Result};
use candle_core::quantized::{ggml_file, gguf_file};
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights;
use futures_util::{Stream, TryStreamExt};
use silent::prelude::{error, SSEEvent};
use silent::{SilentError, StatusCode};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

#[derive(Clone, Debug)]
pub(crate) struct ChatModel {
    tokenizer: Tokenizer,
    model: ModelWeights,
    device: Device,
    seed: u64,
    eos_token: u32,
    chat_format: ChatFormat,
}

pub(crate) struct ChatModelStream {
    pub(crate) response: ChatCompletionResponse,
    pub(crate) max_tokens: usize,
    pub(crate) index: usize,
    pub(crate) result: String,
    next_token: u32,
    device: Device,
    eos_token: u32,
    all_tokens: Vec<u32>,
    sampled: usize,
    model: ModelWeights,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    is_json: bool,
    start_post_prompt: Instant,
    is_finished: bool,
}

impl Stream for ChatModelStream {
    type Item = silent::Result<SSEEvent>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        if self.index >= self.max_tokens || self.is_finished {
            std::task::Poll::Ready(None)
        } else {
            let chunk: silent::Result<ChatCompletionResponseChunk> = {
                let index_pos = self.response.usage.prompt_tokens + self.index;
                let input = Tensor::new(&[self.next_token], &self.device)
                    .map_err(|e| {
                        SilentError::business_error(
                            StatusCode::BAD_REQUEST,
                            format!("failed to create input tensor: {}", e),
                        )
                    })?
                    .unsqueeze(0)
                    .map_err(|e| {
                        SilentError::business_error(
                            StatusCode::BAD_REQUEST,
                            format!("failed to unsqueeze input tensor: {}", e),
                        )
                    })?;
                let logits = self.model.forward(&input, index_pos).map_err(|e| {
                    SilentError::business_error(
                        StatusCode::BAD_REQUEST,
                        format!("failed to forward model: {}", e),
                    )
                })?;
                let logits = logits.squeeze(0).map_err(|e| {
                    SilentError::business_error(
                        StatusCode::BAD_REQUEST,
                        format!("failed to squeeze logits: {}", e),
                    )
                })?;
                let logits = {
                    let start_at = self.all_tokens.len().saturating_sub(64);
                    candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        1.1,
                        &self.all_tokens[start_at..],
                    )
                    .map_err(|e| {
                        SilentError::business_error(
                            StatusCode::BAD_REQUEST,
                            format!("failed to apply repeat penalty: {}", e),
                        )
                    })?
                };
                let next_token = self.logits_processor.sample(&logits).map_err(|e| {
                    SilentError::business_error(
                        StatusCode::BAD_REQUEST,
                        format!("failed to sample next token: {}", e),
                    )
                })?;
                self.next_token = next_token;
                self.all_tokens.push(next_token);
                let next_str = self.tokenizer.decode(&[next_token], true).map_err(|e| {
                    SilentError::business_error(
                        StatusCode::BAD_REQUEST,
                        format!("failed to decode next token: {}", e),
                    )
                })?;
                self.sampled += 1;
                if self.next_token == self.eos_token {
                    self.is_finished = true;
                    Ok(ChatCompletionResponseChunk::from_response(
                        &self.response,
                        vec![ChatCompletionChoice {
                            finish_reason: Default::default(),
                            index: 0,
                            message: AssistantMessage {
                                content: None,
                                name: None,
                                tool_calls: vec![],
                            },
                        }],
                    ))
                } else {
                    println!("next_str: {}", next_str);
                    self.result.push_str(&next_str);
                    self.index += 1;
                    Ok(ChatCompletionResponseChunk::from_response(
                        &self.response,
                        vec![ChatCompletionChoice {
                            finish_reason: FinishReason::Null,
                            index: 0,
                            message: AssistantMessage {
                                content: Some(next_str),
                                name: None,
                                tool_calls: vec![],
                            },
                        }],
                    ))
                }
            };
            match chunk {
                Ok(chunk) => std::task::Poll::Ready(Some(Ok(
                    SSEEvent::default().data(serde_json::to_string(&chunk).unwrap())
                ))),
                Err(e) => {
                    error!("failed to serialize chunk: {}", e);
                    std::task::Poll::Ready(None)
                }
            }
        }
    }
}

impl ChatModel {
    pub(crate) fn handle(&self, request: ChatCompletionRequest) -> Result<ChatCompletionResponse> {
        let ChatCompletionRequest {
            messages,
            temperature,
            top_p,
            max_tokens,
            ..
        } = request;
        let temperature = temperature.map(|temperature| temperature as f64);
        let top_p = top_p.map(|top_p| top_p as f64);
        let prompt = self.chat_format.format_messages(messages)?;
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?;
        let mut model = self.model.clone();

        let mut logits_processor = LogitsProcessor::new(self.seed, temperature, top_p);
        let start_prompt_processing = std::time::Instant::now();
        let mut response = ChatCompletionResponse::new(request.model.clone());
        let mut choice = ChatCompletionChoice {
            finish_reason: Default::default(),
            index: 0,
            message: AssistantMessage {
                content: Some("".to_string()),
                name: None,
                tool_calls: vec![],
            },
        };
        let mut all_tokens = vec![];
        let mut next_token = {
            let input = Tensor::new(tokens.get_ids(), &self.device)?.unsqueeze(0)?;
            let logits = model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            logits_processor.sample(&logits)?
        };
        all_tokens.push(next_token);
        let prompt_dt = start_prompt_processing.elapsed();
        let mut result = self.tokenizer.decode(&[next_token], true).map_err(E::msg)?;
        let start_post_prompt = std::time::Instant::now();
        let mut sampled = 0;
        for index in 0..max_tokens.unwrap_or(4096) {
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = model.forward(&input, tokens.len() + index)?;
            let logits = logits.squeeze(0)?;
            // let logits = if args.repeat_penalty == 1. {
            //     logits
            // } else {
            //     let start_at = all_tokens.len().saturating_sub(args.repeat_last_n);
            //     candle_transformers::utils::apply_repeat_penalty(
            //         &logits,
            //         args.repeat_penalty,
            //         &all_tokens[start_at..],
            //     )?
            // };
            let logits = {
                let start_at = all_tokens.len().saturating_sub(64);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    1.1,
                    &all_tokens[start_at..],
                )?
            };
            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
            let next_str = self.tokenizer.decode(&[next_token], true).map_err(E::msg)?;
            sampled += 1;
            if next_token == self.eos_token {
                break;
            };
            result.push_str(&next_str);
        }
        let dt = start_post_prompt.elapsed();
        println!(
            "\n\n{:4} prompt tokens processed: {:.2} token/s",
            tokens.len(),
            tokens.len() as f64 / prompt_dt.as_secs_f64(),
        );
        println!(
            "{sampled:4} tokens generated: {:.2} token/s",
            sampled as f64 / dt.as_secs_f64(),
        );
        choice.message.content = Some(result);
        response.choices.push(choice);
        response.usage.prompt_tokens = tokens.len();
        response.usage.completion_tokens = sampled;
        response.usage.total_tokens = tokens.len() + sampled;
        Ok(response)
    }
    pub(crate) fn stream_handle(&self, request: ChatCompletionRequest) -> Result<ChatModelStream> {
        let ChatCompletionRequest {
            messages,
            temperature,
            top_p,
            max_tokens,
            response_format,
            ..
        } = request;
        let is_json = match response_format.clone() {
            None => true,
            Some(format) => {
                if format.r#type == ChatResponseFormat::Json {
                    true
                } else {
                    false
                }
            }
        };
        let temperature = temperature.map(|temperature| temperature as f64);
        let top_p = top_p.map(|top_p| top_p as f64);
        let prompt = self.chat_format.format_messages(messages)?;
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?;
        let mut model = self.model.clone();

        let mut logits_processor = LogitsProcessor::new(self.seed, temperature, top_p);
        let start_prompt_processing = std::time::Instant::now();
        let mut all_tokens = vec![];
        let next_token = {
            let input = Tensor::new(tokens.get_ids(), &self.device)?.unsqueeze(0)?;
            let logits = model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            logits_processor.sample(&logits)?
        };
        all_tokens.push(next_token);
        let prompt_dt = start_prompt_processing.elapsed();
        println!(
            "\n\n{:4} prompt tokens processed: {:.2} token/s",
            tokens.len(),
            tokens.len() as f64 / prompt_dt.as_secs_f64(),
        );
        let result = self.tokenizer.decode(&[next_token], true).map_err(E::msg)?;
        let start_post_prompt = std::time::Instant::now();
        let sampled = 0;
        let mut response = ChatCompletionResponse::new(request.model.clone());
        let mut choice = ChatCompletionChoice {
            finish_reason: Default::default(),
            index: 0,
            message: AssistantMessage {
                content: Some(result.clone()),
                name: None,
                tool_calls: vec![],
            },
        };
        choice.message.content = Some(result);
        response.choices.push(choice);
        response.usage.prompt_tokens = tokens.len();
        response.usage.completion_tokens = sampled;
        response.usage.total_tokens = tokens.len() + sampled;
        Ok(ChatModelStream {
            response: response.clone(),
            max_tokens: max_tokens.unwrap_or(4096),
            index: 0,
            result: "".to_string(),
            next_token,
            device: self.device.clone(),
            eos_token: self.eos_token,
            all_tokens,
            sampled,
            model,
            tokenizer: self.tokenizer.clone(),
            logits_processor,
            is_json,
            start_post_prompt,
            is_finished: false,
        })
    }
}

pub(crate) fn init_model(args: ChatModelConfig) -> Result<ChatModel> {
    let ChatModelConfig {
        model_id,
        tokenizer,
        cpu,
        seed,
        gqa,
        chat_format,
        ..
    } = args;
    let device = device(cpu)?;
    // let model_path = args.model_id;
    let model_path = PathBuf::from(model_id);
    let tokenizer_filename = match tokenizer {
        None => match model_path.is_dir() {
            true => PathBuf::from(format!("{:?}/tokenizer.json", model_path)),
            false => PathBuf::from(format!("{:?}/tokenizer.json", model_path.parent().unwrap())),
        },
        Some(tokenizer) => PathBuf::from(tokenizer),
    };

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let mut file = std::fs::File::open(&model_path)?;
    let start = std::time::Instant::now();

    let eos_token = token_id(&tokenizer, &chat_format.get_eos_token())?;
    println!("eos_token: {}", eos_token);

    let model = match model_path.extension().and_then(|v| v.to_str()) {
        Some("gguf") => {
            let model = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
            let mut total_size_in_bytes = 0;
            for (_, tensor) in model.tensor_infos.iter() {
                let elem_count = tensor.shape.elem_count();
                total_size_in_bytes +=
                    elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
            }
            println!(
                "loaded {:?} tensors ({}) in {:.2}s",
                model.tensor_infos.len(),
                &format_size(total_size_in_bytes),
                start.elapsed().as_secs_f32(),
            );
            ModelWeights::from_gguf(model, &mut file, &device)?
        }
        Some("ggml" | "bin") | Some(_) | None => {
            let model = ggml_file::Content::read(&mut file, &device)
                .map_err(|e| e.with_path(model_path))?;
            let mut total_size_in_bytes = 0;
            for (_, tensor) in model.tensors.iter() {
                let elem_count = tensor.shape().elem_count();
                total_size_in_bytes +=
                    elem_count * tensor.dtype().type_size() / tensor.dtype().block_size();
            }
            println!(
                "loaded {:?} tensors ({}) in {:.2}s",
                model.tensors.len(),
                &format_size(total_size_in_bytes),
                start.elapsed().as_secs_f32(),
            );
            println!("params: {:?}", model.hparams);
            ModelWeights::from_ggml(model, gqa, &device)?
        }
    };
    Ok(ChatModel {
        tokenizer,
        model,
        device,
        seed,
        eos_token,
        chat_format,
    })
}
#[cfg(test)]
mod tests {
    use crate::models::chat::init_model;
    use crate::types::chat::ChatCompletionRequest;
    use crate::Config;

    #[test]
    fn chat_test() {
        let config = Config::load("test_chat_config.toml".to_string()).unwrap();
        let config = config.chat_configs.unwrap().pop().unwrap();
        println!("{:?}", config);
        let model = init_model(config).unwrap();
        let json_str = r#"{
    "model": "yi-chat-6b.Q5_K_M.gguf",
    "messages": [
        {
            "role": "system",
            "content": "你是一个有用的AI助手!"
        },
        {
            "role": "user",
            "content": "你好！"
        },
        {
            "role": "assistant",
            "content": "你好！"
        },
        {
            "role": "user",
            "content": "你是谁？"
        }
    ]
}"#;
        let request = serde_json::from_str::<ChatCompletionRequest>(json_str).unwrap();
        let result = model.handle(request).unwrap();
        println!("result: {:?}", result);
    }
}
