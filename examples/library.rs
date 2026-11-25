//! Run with: cargo run --no-default-features --example library -- TRAIN [MODEL]
use serenade::{Algorithm, ModelConfig, Recommender, VMISIndex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(2..=3).contains(&args.len()) {
        return Err("usage: library TRAIN [HPO_MODEL.toml]".into());
    }
    let index = VMISIndex::load(&args[1])?;
    let vmis = if let Some(path) = args.get(2) {
        Recommender::from_config(&index, path)?
    } else {
        Recommender::new(&index, ModelConfig::for_algorithm(Algorithm::Vmis))?
    };
    let wsknn = Recommender::new(&index, ModelConfig::for_algorithm(Algorithm::Wsknn))?;
    let session = &[13598, 2835, 12068];
    for (name, model) in [("configured", vmis), ("wsknn", wsknn)] {
        for item in model.predict(session, 20)? {
            println!("{name}\t{}\t{}", item.id, item.score);
        }
    }
    Ok(())
}
