use clap::Parser;
use llm_server::{get_routes, Args, Config as LlmConfig, Models};
use silent::middlewares::{Cors, CorsType};
use silent::prelude::{logger, Level, Route, Server};
use silent::Configs;

#[tokio::main]
async fn main() {
    logger::fmt().with_max_level(Level::INFO).init();
    let args = Args::parse();
    let mut configs = Configs::default();
    let llm_config = LlmConfig::load(args.configs).expect("failed to load config");
    let host = args.host.unwrap_or(
        llm_config
            .host
            .clone()
            .unwrap_or_else(|| "localhost".to_string()),
    );
    let port = args.port.unwrap_or(llm_config.port.unwrap_or(8000));
    let models = Models::new(llm_config).expect("failed to initialize models");
    configs.insert(models);
    let route = Route::new("").append(get_routes()).hook(
        Cors::new()
            .origin(CorsType::Any)
            .methods(CorsType::Any)
            .headers(CorsType::Any)
            .credentials(false),
    );
    Server::new()
        .with_configs(configs)
        .bind(format!("{host}:{port}").parse().unwrap())
        .serve(route)
        .await
}
