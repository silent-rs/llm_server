use clap::Parser;
use llm_server::types::RequestTypes;
use llm_server::{get_routes, Args, Config as LlmConfig, Models};
use silent::prelude::{logger, Level, Route, Server};
use silent::Configs;
use tokio::sync::mpsc::channel;

#[tokio::main]
async fn main() {
    logger::fmt().with_max_level(Level::INFO).init();
    let args = Args::parse();
    let mut configs = Configs::default();
    let llm_config = LlmConfig::load(args.configs).expect("failed to load config");
    let (tx, rx) = channel::<RequestTypes>(50);
    let host = llm_config
        .host
        .clone()
        .unwrap_or(args.host.clone().unwrap_or_else(|| "localhost".to_string()));
    let port = llm_config.port.clone().unwrap_or(args.port.unwrap_or(8000));
    let mut models = Models::new(llm_config, rx).expect("failed to initialize models");
    configs.insert(tx);
    let route = Route::new("").append(get_routes());
    tokio::spawn(async move {
        Server::new()
            .with_configs(configs)
            .bind(format!("{host}:{port}").parse().unwrap())
            .serve(route)
            .await
    });
    models.handle().await.expect("failed to handle requests");
}
