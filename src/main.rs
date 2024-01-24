use clap::Parser;
use llm_server::types::RequestTypes;
use llm_server::{get_routes, Args, Config as LlmConfig, Models};
use silent::middlewares::{Cors, CorsType};
use silent::prelude::{logger, Level, Route, Server};
use silent::Configs;
use std::any::TypeId;
use std::sync::Arc;
use tokio::sync::mpsc::channel;
use tokio::sync::RwLock;

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
    let port = llm_config.port.unwrap_or(args.port.unwrap_or(8000));
    let mut models = Models::new(llm_config, rx).expect("failed to initialize models");
    configs.insert(tx);
    let route = Route::new("").append(get_routes()).hook(
        Cors::new()
            .origin(CorsType::Any)
            .methods(CorsType::Any)
            .headers(CorsType::Any)
            .credentials(false),
    );
    tokio::spawn(async move {
        Server::new()
            .with_configs(configs)
            .bind(format!("{host}:{port}").parse().unwrap())
            .serve(route)
            .await
    });
    models.handle().await.expect("failed to handle requests");
}
