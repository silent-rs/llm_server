use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
/// LLM server
pub struct Args {
    /// host
    pub host: Option<String>,
    /// port
    pub port: Option<u16>,
    /// config file path
    #[arg(long)]
    pub configs: String,
}
