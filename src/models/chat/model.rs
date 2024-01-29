use crate::configs::ChatModelConfig;
use crate::models::chat::chat_format::ChatFormat;
use crate::models::chat::utils::format_size;
use crate::models::device::{device, token_id};
use crate::types::chat::completion::{AssistantMessage, ChatCompletionChoice};
use crate::types::chat::{
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChunk, ChatResponse,
};
use anyhow::{Error as E, Result};
use candle_core::quantized::{ggml_file, gguf_file};
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights;
use std::io::Write;
use std::path::PathBuf;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::UnboundedSender;

#[derive(Clone, Debug)]
pub(crate) struct ChatModel {
    tokenizer: Tokenizer,
    model: ModelWeights,
    device: Device,
    seed: u64,
    eos_token: u32,
    chat_format: ChatFormat,
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
    pub(crate) fn stream_handle(
        &self,
        request: ChatCompletionRequest,
        tx: UnboundedSender<ChatResponse>,
    ) -> Result<ChatCompletionResponse> {
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
        tx.send(
            ChatCompletionResponseChunk::from_response(&response, vec![choice.clone()]).into(),
        )?;
        for index in 0..max_tokens.unwrap_or(4096) {
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = model.forward(&input, tokens.len() + index)?;
            let logits = logits.squeeze(0)?;
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
            tx.send(
                ChatCompletionResponseChunk::from_response(
                    &response,
                    vec![ChatCompletionChoice {
                        finish_reason: Default::default(),
                        index: 0,
                        message: AssistantMessage {
                            content: Some(next_str),
                            name: None,
                            tool_calls: vec![],
                        },
                    }],
                )
                .into(),
            )?;
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
        tx.send(response.clone().into())?;
        Ok(response)
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

    let mut model = match model_path.extension().and_then(|v| v.to_str()) {
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
