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
    executor: Option<web::Data<crate::executor::BlockingExecutor>>,
) -> HttpResponse {
    if let Some(executor) = executor {
        let query = query.into_inner();
        let result = match executor.submit(move || recommend(&data, query)) {
            Ok(result) => result,
            Err(_) => return HttpResponse::ServiceUnavailable().finish(),
        };
        match result.await {
            Ok(items) => HttpResponse::Ok().json(items),
            Err(_) => HttpResponse::ServiceUnavailable().finish(),
        }
    } else {
        HttpResponse::Ok().json(recommend(&data, query.into_inner()))
    }
}

fn recommend(data: &SharedHandlesAndConfig, query: V1QueryParams) -> Vec<u64> {
    let most_recent_item = query.item_id;
    let user_consent = query.user_consent;
    let session_id_digest = md5::compute(&query.session_id);
    let evolving_session_id = Builder::from_bytes(session_id_digest.0).build().as_u128();

    let vmis_index = data.vmis_index.as_ref();
    let session_store = data.session_store.as_ref();

    let k = data.neighborhood_size_k;
    let m = data.m_most_recent_sessions;
    let how_many = data.num_items_to_recommend;
    let max_items_in_session = data.max_items_in_session;
    let enable_business_logic = data.enable_business_logic;

    let session_items = if user_consent {
        let stored = session_store.try_get_session_items(&evolving_session_id);
        let read_ok = stored.is_ok();
        let mut session_items = stored.unwrap_or_else(|error| {
            eprintln!("session read failed: {:#}", error);
            Vec::new()
        });
        if session_items.is_empty() {
            session_items.push(most_recent_item);
        } else if data.knn.algorithm != vmisknn::algorithms::Algorithm::Vmis
            || session_items.last().unwrap() != &most_recent_item
        {
            session_items.push(most_recent_item);
            if session_items.len() > max_items_in_session {
                // Reduce the amount of session_items to max_items_in_session.
                session_items.drain(0..1);
            }
        }
        if read_ok {
            if let Err(error) =
                session_store.try_update_session_items(&evolving_session_id, &session_items)
            {
                eprintln!("session write failed: {:#}", error);
                session_items = vec![most_recent_item];
            }
        }
        session_items
    } else {
        vec![most_recent_item]
    };

    let recommendations = vmisknn::predict_with_model(
        vmis_index,
        &session_items,
        k,
        m,
        how_many,
        enable_business_logic,
        &data.knn,
        chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
    );

    let recommended_items: Vec<u64> = recommendations
        .into_sorted_vec()
        .iter()
        .map(|scored| scored.id)
        .collect();

    recommended_items
}
