//! Persist the shared index without changing the recommendation server.
use serenade::{config::AppConfig, vmisknn::vmis_index::VMISIndex};
fn main() {
    let config_path = std::env::args()
        .nth(1)
        .expect("usage: build_index CONFIG OUTPUT.vmis [training-stats.bin]");
    let output = std::env::args()
        .nth(2)
        .expect("usage: build_index CONFIG OUTPUT.vmis [training-stats.bin]");
    let config = AppConfig::new(config_path);
    let index = VMISIndex::load_for_model(
        &config.data.training_data_path,
        config.model.m_most_recent_sessions,
        config.model.idf_weighting,
        &config.model.knn,
    )
    .expect("Cannot construct selected kNN model");
    if let Some(metadata) = std::env::args().nth(3) {
        index
            .save_training_statistics(std::path::Path::new(&metadata))
            .expect("Cannot write training statistics");
    }
    index.save(&output).expect("Cannot write VMIS snapshot");
    println!(
        "Saved {} (estimated owned index allocation: {} bytes)",
        output,
        index.estimated_heap_bytes()
    );
}
