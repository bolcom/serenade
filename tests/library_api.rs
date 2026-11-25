use serenade::{Algorithm, ModelConfig, PredictionScratch, Recommendation, Recommender, VMISIndex};
use std::path::PathBuf;

struct Fixture(PathBuf);
impl Fixture {
    fn new(name: &str) -> Self {
        let root =
            std::env::temp_dir().join(format!("serenade-library-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let mut data = String::from("SessionId\tItemId\tTime\n");
        for session in 0..30 {
            for (position, item) in [1, 2 + session % 5, 8 + session % 3, 1].iter().enumerate() {
                data.push_str(&format!(
                    "{session}\t{item}\t{}\n",
                    100 + session * 4 + position
                ));
            }
        }
        std::fs::write(root.join("train.tsv"), data).unwrap();
        Self(root)
    }
    fn train(&self) -> String {
        self.0.join("train.tsv").to_str().unwrap().to_owned()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
fn model(algorithm: Algorithm) -> ModelConfig {
    let mut model = ModelConfig::for_algorithm(algorithm);
    model.m_most_recent_sessions = 20;
    model.neighborhood_size_k = 3;
    model.max_items_in_session = 3;
    model.idf_weighting = 0.75;
    if algorithm == Algorithm::Vsknn {
        model.knn.idf_weighting = 1.75;
    }
    model
}
fn pairs(items: Vec<Recommendation>) -> Vec<(u64, f64)> {
    items
        .into_iter()
        .map(|item| (item.id, item.score))
        .collect()
}

#[test]
fn seven_models_match_application_predictions_share_storage_and_run_concurrently() {
    let fixture = Fixture::new("shared");
    let index = VMISIndex::builder()
        .capacity(30)
        .load(fixture.train())
        .unwrap();
    let before = fixture.0.join("before.vmis");
    index.save(before.to_str().unwrap()).unwrap();
    let models: Vec<_> = Algorithm::ALL
        .into_iter()
        .map(|a| Recommender::new(&index, model(a)).unwrap())
        .collect();
    let memory = index.estimated_heap_bytes();
    let more: Vec<_> = Algorithm::ALL
        .into_iter()
        .map(|a| Recommender::new(&index, model(a)).unwrap())
        .collect();
    assert_eq!(
        memory,
        index.estimated_heap_bytes(),
        "models must reuse item statistics and corpus"
    );
    let after = fixture.0.join("after.vmis");
    index.save(after.to_str().unwrap()).unwrap();
    assert_eq!(
        std::fs::read(before).unwrap(),
        std::fs::read(after).unwrap()
    );
    for (algorithm, recommender) in Algorithm::ALL.into_iter().zip(&models) {
        let config = model(algorithm);
        let direct =
            VMISIndex::load_for_model(&fixture.train(), 20, config.idf_weighting, &config.knn)
                .unwrap();
        for session in [
            &[][..],
            &[999][..],
            &[1][..],
            &[1, 3][..],
            &[7, 1, 2, 1][..],
        ] {
            let tail = &session[session.len().saturating_sub(3)..];
            let expected = serenade::vmisknn::predict_with_model(
                &direct,
                tail,
                3,
                20,
                8,
                false,
                &config.knn,
                225.0,
            )
            .into_sorted_vec();
            let actual = recommender.predict_at(session, 8, 225.0).unwrap();
            assert_eq!(pairs(actual), pairs(expected), "{algorithm:?}");
        }
        assert!(!recommender
            .predict_at(&[1, 3], 8, 225.0)
            .unwrap()
            .is_empty());
        let expected = pairs(recommender.predict_at(&[1, 3], 8, 225.0).unwrap());
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let expected = &expected;
                scope.spawn(move || {
                    let mut scratch = PredictionScratch::default();
                    for _ in 0..20 {
                        assert_eq!(
                            pairs(recommender.predict_at(&[1, 3], 8, 225.0).unwrap()),
                            *expected
                        );
                        assert_eq!(
                            pairs(
                                recommender
                                    .predict_with_scratch(&[1, 3], 8, 225.0, &mut scratch)
                                    .unwrap()
                            ),
                            *expected
                        );
                    }
                });
            }
        });
    }
    drop(more);
}

#[test]
fn model_files_snapshots_and_invalid_inputs() {
    let fixture = Fixture::new("snapshots");
    let index = VMISIndex::builder()
        .capacity(30)
        .load(fixture.train())
        .unwrap();
    let path = fixture.0.join("model.toml");
    std::fs::write(&path, "[model]\nalgorithm=\"vmis\"\nm_most_recent_sessions=20\nneighborhood_size_k=3\nmax_items_in_session=3\nidf_weighting=0.75\n").unwrap();
    let recommender = Recommender::from_config(&index, &path).unwrap();
    let expected = pairs(recommender.predict_at(&[1, 3], 8, 225.0).unwrap());
    assert_eq!(
        expected,
        pairs(
            Recommender::new(&index, model(Algorithm::Vmis))
                .unwrap()
                .predict_at(&[1, 3], 8, 225.0)
                .unwrap()
        )
    );
    for algorithm in [Algorithm::Vmis, Algorithm::VsknnStan] {
        let config = model(algorithm);
        let weighted = VMISIndex::load_for_model(&fixture.train(), 30, 2.75, &config.knn).unwrap();
        let snapshot = fixture.0.join("index.vmis");
        weighted.save(snapshot.to_str().unwrap()).unwrap();
        let restored = VMISIndex::load(&snapshot).unwrap();
        assert!(VMISIndex::builder().capacity(31).load(&snapshot).is_err());
        for algorithm in Algorithm::ALL {
            let direct = Recommender::new(&index, model(algorithm)).unwrap();
            let loaded = Recommender::new(&restored, model(algorithm)).unwrap();
            let older = Recommender::new(&weighted, model(algorithm)).unwrap();
            assert_eq!(
                pairs(direct.predict_at(&[1, 3], 8, 225.0).unwrap()),
                pairs(older.predict_at(&[1, 3], 8, 225.0).unwrap())
            );
            assert_eq!(
                pairs(direct.predict_at(&[1, 3], 8, 225.0).unwrap()),
                pairs(loaded.predict_at(&[1, 3], 8, 225.0).unwrap())
            );
        }
    }
    assert!(VMISIndex::builder()
        .capacity(0)
        .load(fixture.train())
        .is_err());
    assert!(VMISIndex::load(fixture.0.join("missing")).is_err());
    let mut invalid = model(Algorithm::Vmis);
    invalid.m_most_recent_sessions = 31;
    assert!(Recommender::new(&index, invalid.clone()).is_err());
    invalid.m_most_recent_sessions = 20;
    invalid.max_items_in_session = 0;
    assert!(Recommender::new(&index, invalid).is_err());
    assert!(recommender.predict_at(&[1], 8, f64::NAN).is_err());
    assert!(recommender.predict_at(&[1], 8, -1.0).is_err());
    assert!(recommender.predict(&[1], 0).unwrap().is_empty());
    assert!(!recommender.predict(&[1], usize::MAX).unwrap().is_empty());
}
