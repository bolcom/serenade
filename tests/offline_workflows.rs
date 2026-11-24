use serenade::{
    config::ModelConfig,
    offline::{self, EvaluationData},
    vmisknn::{
        algorithms::{Algorithm, KnnConfig},
        predict_with_model,
        vmis_index::VMISIndex,
    },
};

#[test]
fn reused_index_matches_fresh_indexes_and_keeps_holdout_separate() {
    let directory = std::env::temp_dir().join(format!("serenade-offline-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let training = directory.join("train.tsv");
    let holdout = directory.join("holdout.tsv");
    let mut events = String::from("SessionId\tItemId\tTime\n");
    for session in 0..30 {
        for (position, item) in [1, 2 + session % 5, 8 + session % 3, 1].iter().enumerate() {
            events.push_str(&format!(
                "{}\t{}\t{}\n",
                session,
                item,
                100 + session / 2 + position
            ));
        }
    }
    std::fs::write(&training, events).unwrap();
    std::fs::write(
        &holdout,
        "SessionId\tItemId\tTime\n1\t1\t200\n1\t999\t201\n2\t1\t200\n2\t3\t201\n2\t8\t202\n2\t1\t203\n",
    )
    .unwrap();
    let data = EvaluationData::load(holdout.to_str().unwrap()).unwrap();
    let parallel = offline::execution::Execution::new(4).unwrap();
    let sequential = offline::execution::Execution::new(1).unwrap();
    let mut shared = VMISIndex::new_from_csv(training.to_str().unwrap(), 30, 1.0);
    // Visit VMIS twice, restoring its original IDF after reference scoring.
    for algorithm in Algorithm::ALL.into_iter().chain([Algorithm::Vmis]) {
        let mut knn = KnnConfig::for_algorithm(algorithm);
        if algorithm == Algorithm::Vsknn {
            knn.idf_weighting = 1.75;
        }
        for m in [1, 3, 8, 20] {
            shared.prepare_for_model(m, 0.75, &knn).unwrap();
            let fresh =
                VMISIndex::load_for_model(training.to_str().unwrap(), m, 0.75, &knn).unwrap();
            let snapshot = directory.join("index.vmis");
            shared.save(snapshot.to_str().unwrap()).unwrap();
            assert!(VMISIndex::load_corpus(snapshot.to_str().unwrap(), 31).is_err());
            let restored =
                VMISIndex::load_for_model(snapshot.to_str().unwrap(), m, 0.75, &knn).unwrap();
            for current in [&[][..], &[999][..], &[1][..], &[1, 3][..], &[1, 2, 1][..]] {
                let predictions = |index: &VMISIndex| {
                    predict_with_model(index, current, 3, m, 20, false, &knn, 200.0)
                        .into_sorted_vec()
                        .into_iter()
                        .map(|s| (s.id, s.score))
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    predictions(&shared),
                    predictions(&fresh),
                    "{:?}, m={}",
                    algorithm,
                    m
                );
                assert_eq!(predictions(&shared), predictions(&restored));
                assert!(!predictions(&shared).iter().any(|(id, _)| *id == 999));
            }
            let model = ModelConfig {
                knn: knn.clone(),
                m_most_recent_sessions: m,
                neighborhood_size_k: 3,
                max_items_in_session: 2,
                num_items_to_recommend: 20,
                idf_weighting: 0.75,
            };
            // Scoring views scale an existing unit IDF instead of rescanning it.
            shared.prepare_for_model(30, 1.0, &knn).unwrap();
            let before = directory.join("before.vmis");
            shared.save(before.to_str().unwrap()).unwrap();
            for text in ["mrr@1", "ndcg@8", "hitrate@20"] {
                let objective: serenade::metrics::objective::Objective = text.parse().unwrap();
                let expected = offline::evaluate_at_cutoff(
                    &fresh,
                    &data,
                    &model,
                    false,
                    std::num::NonZeroUsize::new(objective.cutoff()).unwrap(),
                )
                .score(objective);
                let one = sequential
                    .run(&shared, &data, &model, false, objective)
                    .unwrap();
                let many = parallel
                    .run(&shared, &data, &model, false, objective)
                    .unwrap();
                assert!(
                    (one.objective - expected).abs() < 1e-12,
                    "{algorithm:?} {text}"
                );
                assert!((many.objective - expected).abs() < 1e-12);
                assert_eq!(many.predictions, 4);
            }
            let after = directory.join("after.vmis");
            shared.save(after.to_str().unwrap()).unwrap();
            assert_eq!(
                std::fs::read(before).unwrap(),
                std::fs::read(after).unwrap()
            );
            let result = offline::evaluate(&shared, &data, &model, false);
            assert_eq!(result.predictions, 4);
        }
    }
    std::fs::remove_dir_all(directory).unwrap();
}
