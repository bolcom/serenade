use crate::vmisknn::vmis_index::VMISIndex;

// objective function
pub fn objective_with_model(
    path_to_training: std::string::String,
    test_data_file: std::string::String,
    n_most_recent_sessions: i32,
    neighborhood_size_k: i32,
    last_items_in_session: i32,
    idf_weighting: f64,
    enable_business_logic: bool,
    model: &crate::vmisknn::algorithms::KnnConfig,
) -> f64 {
    use crate::vmisknn::algorithms::Algorithm;
    let mut model = model.clone();
    match model.algorithm {
        Algorithm::Vsknn => model.idf_weighting = idf_weighting,
        Algorithm::VsknnStan => model.lambda_idf = Some(idf_weighting),
        _ => {}
    }
    let vmis_index = VMISIndex::load_for_model(
        &path_to_training,
        n_most_recent_sessions as usize,
        idf_weighting,
        &model,
    )
    .expect("Cannot load selected kNN model");

    let data = crate::offline::EvaluationData::load(&test_data_file).expect("load objective data");
    let config = crate::config::ModelConfig {
        knn: model,
        m_most_recent_sessions: n_most_recent_sessions as usize,
        neighborhood_size_k: neighborhood_size_k as usize,
        max_items_in_session: last_items_in_session as usize,
        num_items_to_recommend: 20,
        idf_weighting,
    };
    crate::offline::evaluate(&vmis_index, &data, &config, enable_business_logic).mrr_at_20
}
