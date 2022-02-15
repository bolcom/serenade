extern crate sys_info;

use actix_web::{get, web, HttpResponse};
use chrono::Utc;

use crate::dataframeutils::SharedHandlesAndConfig;
use web::Data;

#[get("/internal")]
pub async fn internal(config: Data<SharedHandlesAndConfig>) -> HttpResponse {
    let mut html = "<html>serenade: realtime session based recommendations.<br />".to_string();

    let data_stats = &config.vsknn_index.training_data_stats;
    html.push_str("<h3>Training data</h3>");
    html.push_str("Loaded: ");
    html.push_str(&*data_stats.descriptive_name);
    html.push_str("<br />Qty Training Records: ");
    html.push_str(&*data_stats.qty_records.to_string());
    html.push_str("<br />Qty Unique SessionIds: ");
    html.push_str(&*data_stats.qty_unique_session_ids.to_string());
    html.push_str("<br />Qty Unique ItemIds: ");
    html.push_str(&*data_stats.qty_unique_item_ids.to_string());
    html.push_str("<br />Min Date Time: ");
    html.push_str(&data_stats.min_time_date_time.to_string());
    html.push_str("<br />Max Date Time: ");
    html.push_str(&data_stats.max_time_date_time.to_string());
    html.push_str("<br />Age (hours): ");

    let age_hours = (Utc::now().naive_utc() - data_stats.max_time_date_time).num_hours();

    html.push_str(&*age_hours.to_string());
    html.push_str("<br />Session duration percentiles (secs): ");
    html.push_str(" p5=");
    html.push_str(&data_stats.session_duration_p05.to_string());
    html.push_str(" p25=");
    html.push_str(&data_stats.session_duration_p25.to_string());
    html.push_str(" p50=");
    html.push_str(&data_stats.session_duration_p50.to_string());
    html.push_str(" p75=");
    html.push_str(&data_stats.session_duration_p75.to_string());
    html.push_str(" p90=");
    html.push_str(&data_stats.session_duration_p90.to_string());
    html.push_str(" p95=");
    html.push_str(&data_stats.session_duration_p95.to_string());
    html.push_str(" p99=");
    html.push_str(&data_stats.session_duration_p99.to_string());
    html.push_str(" p99.5=");
    html.push_str(&data_stats.session_duration_p99_5.to_string());
    html.push_str(" p100=");
    html.push_str(&data_stats.session_duration_p100.to_string());

    html.push_str("<br />Session qty events percentiles: ");
    html.push_str(" p5=");
    html.push_str(&data_stats.qty_events_p05.to_string());
    html.push_str(" p25=");
    html.push_str(&data_stats.qty_events_p25.to_string());
    html.push_str(" p50=");
    html.push_str(&data_stats.qty_events_p50.to_string());
    html.push_str(" p75=");
    html.push_str(&data_stats.qty_events_p75.to_string());
    html.push_str(" p90=");
    html.push_str(&data_stats.qty_events_p90.to_string());
    html.push_str(" p95=");
    html.push_str(&data_stats.qty_events_p95.to_string());
    html.push_str(" p99=");
    html.push_str(&data_stats.qty_events_p99.to_string());
    html.push_str(" p99.5=");
    html.push_str(&data_stats.qty_events_p99_5.to_string());
    html.push_str(" p100=");
    html.push_str(&data_stats.qty_events_p100.to_string());

    html.push_str("<h3>Models</h3>");
    html.push_str("hyperparameters");
    html.push_str("<br />m : ");
    html.push_str(&config.m_most_recent_sessions.to_string());
    html.push_str(" (most_recent_neighbors for evolving session)");
    html.push_str("<br />k : ");
    html.push_str(&config.neighborhood_size_k.to_string());
    html.push_str(" (top `k` closest_neighbor sessions for item scoring)");
    html.push_str("<br />Max items in evolving session:");
    html.push_str(&config.max_items_in_session.to_string());
    html.push_str("<br />Qty items to recommend: ");
    html.push_str(&config.num_items_to_recommend.to_string());
    html.push_str("<br /><a href=\"/v1/recommend?session_id=144&user_consent=true&item_id=1001004010971015\">v1 endpoint of our model</a>");
    html.push_str("<h3>Machine instance</h3>");
    html.push_str("<br />Qty CPU's detected: ");
    html.push_str(&*sys_info::cpu_num().unwrap_or(0).to_string());
    html.push_str("<br />Qty actix workers set: ");
    html.push_str(&config.qty_workers.to_string());
    html.push_str("<br />CPU speed: ");
    html.push_str(&*sys_info::cpu_speed().unwrap_or(0).to_string());
    html.push_str("MHz");
    html.push_str("<br />Active processes on instance: ");
    html.push_str(&*sys_info::proc_total().unwrap_or(0).to_string());
    html.push_str("<h3>Session store</h3>");
    html.push_str("<br />Compaction TTL: ");
    html.push_str(&*config.db_compaction_ttl_in_secs.to_string());
    html.push_str(" seconds");
    html.push_str("<h3>Metrics</h3>");
    html.push_str("<a href=\"/internal/prometheus\">prometheus</a>");
    html.push_str("</html>");

    HttpResponse::Ok().body(html)
}
