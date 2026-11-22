use crate::{io, vmisknn};

use crate::metrics::mrr::Mrr;
use crate::metrics::SessionMetric;
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

    let ordered_test_sessions = io::read_test_data_timed(&*test_data_file);

    let qty_max_reco_results = 20;
    let mut mymetric = Mrr::new(qty_max_reco_results);

    ordered_test_sessions
        .iter()
        .for_each(|(_session_id, events)| {
            let evolving_session_items: Vec<_> = events.iter().map(|(item, _)| *item).collect();
            for session_state in 1..evolving_session_items.len() {
                // use last x items of evolving session
                let start_index = if session_state > last_items_in_session as usize {
                    session_state - last_items_in_session as usize
                } else {
                    0
                };
                let session: &[u64] = &evolving_session_items[start_index..session_state];
                let recommendations = vmisknn::predict_with_model(
                    &vmis_index,
                    &session,
                    neighborhood_size_k as usize,
                    n_most_recent_sessions as usize,
                    qty_max_reco_results,
                    enable_business_logic,
                    &model,
                    events[session_state - 1].1,
                );
                let recommended_items = recommendations
                    .into_sorted_vec()
                    .iter()
                    .map(|scored| scored.id)
                    .collect::<Vec<u64>>();
                let actual_next_items = &evolving_session_items[session_state..];
                mymetric.add(&recommended_items, actual_next_items);
            }
        });
    return mymetric.result();
}
