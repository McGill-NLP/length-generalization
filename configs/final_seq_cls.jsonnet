{
    trainer+: {
        eval_steps: 2000,
        callbacks: [
            (import 'trainer/callbacks/save_predictions.jsonnet'),
            (import 'trainer/callbacks/save_predictions_test.jsonnet'),
        ],
    },
}
