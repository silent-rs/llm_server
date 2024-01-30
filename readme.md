# LLM Server

[Read this in English](./readme_en.md)

## 描述

LLM Server 是一个使用Rust开发，基于 [silent](https://github.com/hubertshelley/silent)
和 [candle](https://github.com/huggingface/candle) 的大语言模型服务，提供了类似openai的接口，易于部署和使用。

## 目前支持的模型

- [whisper](https://github.com/openai/whisper)
- llama及其衍生模型的gguf量化版本

## 安装

使用本服务默认你已经安装了Rust

```shell
# 获取代码
git clone https://github.com/silent-rs/llm_server.git
cd llm_server
# 编译
cargo build --release
# 开启Mac Metal支持
cargo build --release --features metal
# 开启CUDA支持 (需要安装CUDA驱动)
cargo build --release --features cuda
# 运行程序
./target/release/llm_server --configs ./configs.toml

```

## 配置文件说明

```toml
# 服务监听地址，优先获取运行参数host, 其次获取环境变量HOST，再次获取配置文件，最后使用默认值localhost
host = "0.0.0.0"
# 服务监听端口，优先获取运行参数port, 其次获取环境变量PORT，再次获取配置文件，最后使用默认值8000
port = 8000
# 对话模型配置列表
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

# 语音转文字模型配置列表
[[whisper_configs]]
model_id = "model_path/whisper-large-v3"
# alias 为模型的别名，用于区分不同的模型，目前不同模型的别名不能相同且固定
alias = "large-v3"
cpu = false
```