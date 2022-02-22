use crate::{io, vmisknn};

use crate::vmisknn::vmis_index::OfflineIndex;
use crate::metrics::mrr::Mrr;
use crate::metrics::SessionMetric;

// objective function
pub fn objective(
    path_to_training: std::string::String, 
    test_data_file: std::string::String, 
    n_most_recent_sessions: i32, 
    neighborhood_size_k: i32, 
    last_items_in_session: i32,
    enable_business_logic: bool) -> f64 {
    
    let offline_index = OfflineIndex::new_from_csv(&*path_to_training, n_most_recent_sessions as usize);

    let ordered_test_sessions = io::read_test_data_evolving(&*test_data_file);

    let qty_max_reco_results = 20;
    let mut mymetric = Mrr::new(qty_max_reco_results);

    ordered_test_sessions
    .iter()
    .for_each(|(_session_id, evolving_session_items)| {
        for session_state in 1..evolving_session_items.len() {
            // use last x items of evolving session
            let start_index = if session_state > last_items_in_session as usize {
                session_state - last_items_in_session as usize
            } else {
                0
            };
            let session: &[u64] = &evolving_session_items[start_index..session_state];
            let recommendations = vmisknn::predict(
                &offline_index,
                &session,
                neighborhood_size_k as usize,
                n_most_recent_sessions as usize,
                qty_max_reco_results,
                enable_business_logic
            );
            let recommended_items = recommendations
                .into_sorted_vec()
                .iter()
                .map(|scored| scored.id)
                .collect::<Vec<u64>>();
            let actual_next_items = Vec::from(&evolving_session_items[session_state..]);
            mymetric.add(&recommended_items, &actual_next_items);
        }            
    });
    return mymetric.result();
}