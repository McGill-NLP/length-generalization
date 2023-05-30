{
    trainer+: {
        callbacks: [
            (import 'trainer/callbacks/save_predictions.jsonnet'),
//            (import 'trainer/callbacks/save_predictions_test.jsonnet'),
//            (import 'trainer/callbacks/cartography_measure_valid.jsonnet') + { label_index: $.dataset.answer_index_in_label },
//            (import 'trainer/callbacks/cartography_measure_test.jsonnet') + { label_index: $.dataset.answer_index_in_label },
        ],
    },
}
