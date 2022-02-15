use actix_web::{get, web, HttpResponse};
use serde::Deserialize;

use uuid::Builder;

use crate::dataframeutils::SharedHandlesAndConfig;
use crate::vmisknn;

#[derive(Debug, Deserialize)]
pub struct V1QueryParams {
    item_id: u64,
    session_id: String,
    user_consent: bool,
}

// Serenade's main endpoint.
// This endpoint requires GET query parameters because the istio uses the same `session_id` query param for pod affinity.
// This minimizes the risk that the istio uses a different session_id value from the X-header than we use on the GET request.
// There are multiple session_id's for a visitor during a visit (jsession_id, measuring_session_id, etc).
#[get("/v1/recommend")]
pub async fn v1_recommend(
    data: web::Data<SharedHandlesAndConfig>,
    query: web::Query<V1QueryParams>,
) -> HttpResponse {
    let most_recent_item = query.item_id;
    let user_consent = query.user_consent;
    let session_id_digest = md5::compute(&query.session_id);
    let evolving_session_id = Builder::from_bytes(session_id_digest.0).build().as_u128();

    let vsknn_index = data.vsknn_index.as_ref();
    let session_store = data.session_store.as_ref();

    let k = data.neighborhood_size_k;
    let m = data.m_most_recent_sessions;
    let how_many = data.num_items_to_recommend;
    let max_items_in_session = data.max_items_in_session;
    let enable_business_logic = data.enable_business_logic;

    let session_items = if user_consent {
        let mut session_items = session_store.get_session_items(&evolving_session_id);
        if session_items.is_empty() {
            session_items.push(most_recent_item);
        } else if session_items.last().unwrap() != &most_recent_item {
            session_items.push(most_recent_item);
            if session_items.len() > max_items_in_session {
                // Reduce the amount of session_items to max_items_in_session.
                session_items.drain(0..1);
            }
        }
        session_store.update_session_items(&evolving_session_id, &session_items);
        session_items
    } else {
        vec![most_recent_item]
    };

    let recommendations = vmisknn::predict(vsknn_index, &session_items, k, m, how_many, enable_business_logic);

    let recommended_items: Vec<u64> = recommendations
        .into_sorted_vec()
        .iter()
        .map(|scored| scored.id)
        .collect();

    HttpResponse::Ok().json(recommended_items)
}
