use sessions::RocksDBSessionStore;

use actix_web::{
    http::ContentEncoding, middleware, web, App, HttpRequest, HttpResponse, HttpServer,
};
use actix_web_prom::PrometheusMetrics;

use actix_web::http::header;
use std::sync::Arc;
use std::time::Duration;

use serenade::config::AppConfig;
use serenade::dataframeutils::SharedHandlesAndConfig;
use serenade::endpoints::index_resource::internal;
use serenade::endpoints::recommend_resource::v1_recommend;
use serenade::sessions;
use serenade::vmisknn::vmis_index::VMISIndex;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let config_path = std::env::args().nth(1).unwrap_or_default();
    let config = AppConfig::try_new(config_path)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;

    let bind_address = format!("{}:{}", config.server.host, config.server.port);
    let m_most_recent_sessions = config.model.m_most_recent_sessions;
    let neighborhood_size_k = config.model.neighborhood_size_k;
    let num_items_to_recommend = config.model.num_items_to_recommend;
    let max_items_in_session = config.model.max_items_in_session;
    let qty_workers = config.server.num_workers;
    let enable_business_logic = config.logic.enable_business_logic;
    let example_item_id = config.logic.example_item_id;

    let vmis_index = Arc::new(
        VMISIndex::load_for_model(
            &config.data.training_data_path,
            config.model.m_most_recent_sessions,
            config.model.idf_weighting,
            &config.model.knn,
        )
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?,
    );
    let knn = config.model.knn.clone();

    println!("start db");
    let session_ttl = Duration::from_secs(30 * 60);
    let db = Arc::new(
        RocksDBSessionStore::try_new("./sessions.db", session_ttl)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error))?,
    );

    // Inline remains the default until representative benchmarks justify offloading.
    let executor = match std::env::var("PREDICTION_EXECUTOR").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("inline") => None,
        Ok("blocking") => {
            let positive = |key: &str, default: usize| -> std::io::Result<usize> {
                let value = match std::env::var(key) {
                    Ok(value) => value.parse::<usize>().map_err(|error| {
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            format!("{}: {}", key, error),
                        )
                    })?,
                    Err(std::env::VarError::NotPresent) => default,
                    Err(error) => {
                        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, error))
                    }
                };
                if value == 0 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!("{} must be positive", key),
                    ));
                }
                Ok(value)
            };
            let threads = positive("PREDICTION_THREADS", qty_workers)?;
            let capacity = positive("PREDICTION_QUEUE_CAPACITY", threads.saturating_mul(2))?;
            Some(web::Data::new(
                serenade::executor::BlockingExecutor::new(threads, capacity)
                    .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error))?,
            ))
        }
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "PREDICTION_EXECUTOR must be inline or blocking",
            ))
        }
    };

    println!("start metrics");
    let prometheus = PrometheusMetrics::new("api", Some("/internal/prometheus"), None);

    println!("Done. start httpd at http://{}", &bind_address);
    HttpServer::new(move || {
        let handles_and_config = SharedHandlesAndConfig {
            knn: knn.clone(),
            session_store: db.clone(),
            vmis_index: vmis_index.clone(),
            m_most_recent_sessions,
            neighborhood_size_k,
            num_items_to_recommend,
            max_items_in_session,
            qty_workers,
            db_compaction_ttl_in_secs: session_ttl.as_secs() as usize,
            enable_business_logic,
            example_item_id,
        };

        let mut app = App::new().data(handles_and_config);
        if let Some(executor) = &executor {
            app = app.app_data(executor.clone());
        }
        app.wrap(middleware::Compress::new(ContentEncoding::Identity))
            .wrap(prometheus.clone())
            .wrap(
                middleware::DefaultHeaders::new()
                    .header("Cache-Control", "no-cache, no-store, must-revalidate")
                    .header("Pragma", "no-cache")
                    .header("Expires", "0"),
            )
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
    .map_err(|error| {
        std::io::Error::new(
            error.kind(),
            format!("bind server to {}: {}", bind_address, error),
        )
    })?
    .run()
    .await
}
