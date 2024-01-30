# LLM Server

[中文说明](./readme.md)

## describe

LLM Server is developed using Rust and based on [silent](https://github.com/hubertshelley/silent)
And [candle](https://github.com/huggingface/candle)'s large language model service provides an openai-like interface
that is easy to deploy and use.

## Currently supported models

- [whisper](https://github.com/openai/whisper)
- gguf quantized version of llama and its derived models

## Install

By default, you have installed Rust when using this service.

```shell
# Get code
git clone https://github.com/silent-rs/llm_server.git
cd llm_server
# compile
cargo build --release
# Enable Mac Metal support
cargo build --release --features metal
# Enable CUDA support (requires CUDA driver installation)
cargo build --release --features cuda
# Run the program
./target/release/llm_server --configs ./configs.toml

```

## Configuration file description

```toml
# Service listening address, first get the running parameter host, then get the environment variable HOST, then get the configuration file, and finally use the default value localhost
host = "0.0.0.0"
# Service listening port, first get the running parameter port, then get the environment variable PORT, then get the configuration file, and finally use the default value 8000
port = 8000
# Dialog model configuration list
[[chat_configs]]
model_id = "model_path/yi-chat-6b.Q5_K_M.gguf"
alias = "yi-chat-6b.Q5_K_M.gguf"
cpu = false
gqa = 1
tokenizer = "model_path/tokenizer.json"
[[chat_configs]]
model_id = "model_path/yi-chat-6b.Q5.gguf"
alias = "yi-chat-6b.Q5.gguf"
cpu = false
gqa = 1
tokenizer = "model_path/tokenizer.json"

# Speech-to-text model configuration list
[[whisper_configs]]
model_id = "model_path/whisper-large-v3"
#alias is the alias of the model, used to distinguish different models. Currently, the aliases of different models cannot be the same and fixed.
alias = "large-v3"
cpu = false
```