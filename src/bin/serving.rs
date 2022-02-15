extern crate serenade_optimized;

use sessions::RocksDBSessionStore;

use actix_web::{
    http::ContentEncoding, middleware, web, App, HttpRequest, HttpResponse, HttpServer,
};
use actix_web_prom::PrometheusMetrics;

use actix_web::http::header;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use serenade_optimized::config::AppConfig;
use serenade_optimized::dataframeutils::SharedHandlesAndConfig;
use serenade_optimized::endpoints::index_resource::internal;
use serenade_optimized::endpoints::recommend_resource::v1_recommend;
use serenade_optimized::sessions;
use serenade_optimized::vmisknn::offline_index::OfflineIndex;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config_path = std::env::args().nth(1).unwrap_or_default();
    let config = AppConfig::new(config_path);

    let bind_address = format!("{}:{}", config.server.host, config.server.port);
    let m_most_recent_sessions = config.model.m_most_recent_sessions;
    let neighborhood_size_k = config.model.neighborhood_size_k;
    let num_items_to_recommend = config.model.num_items_to_recommend;
    let max_items_in_session = config.model.max_items_in_session;
    let qty_workers = config.server.num_workers;
    let enable_business_logic = config.logic.enable_business_logic;

    let training_data_path = Path::new(&config.data.training_data_path);
    let vsknn = if training_data_path.is_dir() {
        // By default we use an index that is computed offline on billions of user-item interactions.
        Arc::new(OfflineIndex::new(&config.data.training_data_path))
    } else if training_data_path.is_file() {
        // The following line creates an index directly from a csv file as input.
        Arc::new(OfflineIndex::new_from_csv(
            &config.data.training_data_path,
            config.model.m_most_recent_sessions,
        ))
    } else {
        panic!(
            "Training data file does not exist: {}",
            &config.data.training_data_path
        )
    };

    println!("start db");
    let session_ttl = Duration::from_secs(30 * 60);
    let db = Arc::new(RocksDBSessionStore::new("./sessions.db", session_ttl));

    println!("start metrics");
    let prometheus = PrometheusMetrics::new("api", Some("/internal/prometheus"), None);

    println!("Done. start httpd at http://{}", &bind_address);
    HttpServer::new(move || {
        let handles_and_config = SharedHandlesAndConfig {
            session_store: db.clone(),
            vsknn_index: vsknn.clone(),
            m_most_recent_sessions,
            neighborhood_size_k,
            num_items_to_recommend,
            max_items_in_session,
            qty_workers,
            db_compaction_ttl_in_secs: session_ttl.as_secs() as usize,
            enable_business_logic,
        };

        App::new()
            .wrap(middleware::Compress::new(ContentEncoding::Identity))
            .wrap(prometheus.clone())
            .wrap(
                middleware::DefaultHeaders::new()
                    .header("Cache-Control", "no-cache, no-store, must-revalidate")
                    .header("Pragma", "no-cache")
                    .header("Expires", "0"),
            )
            .data(handles_and_config)
            .service(v1_recommend)
            .service(internal)
            .service(web::resource("/").route(web::get().to(|_req: HttpRequest| {
                HttpResponse::Found()
                    .header(header::LOCATION, "/internal")
                    .finish()
            })))
    })
    .workers(config.server.num_workers)
    .bind(&bind_address)
    .unwrap_or_else(|_| panic!("Could not bind server to address {}", &bind_address))
    .run()
    .await
}
