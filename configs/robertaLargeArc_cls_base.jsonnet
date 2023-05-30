(import 'base.jsonnet')
+ (import 'models/roberta.jsonnet')
+ (import 'models/roberta_large.jsonnet')
+ (import 'tokenizers/pretrained_fast.jsonnet')
+ (import 'trainer/base_model.jsonnet')
+ (import 'data/seq_cls.jsonnet')
+ (import 'seq_classification.jsonnet')
+ {
    global_vars+: {
        debug_mode: false,
    },
    dataset+: {
        is_encoder_only: true,
        is_decoder_only: false,
    },
    model+: {
        from_pretrained: false,
    },
    trainer+: {
        type: 'trainer_with_metrics',
        metric_for_best_model: if $.dataset.name == 'sum' then 'mse' else 'accuracy',
    },
    analyzers: [
        (import 'analyzers/seq_cls_analyzer.jsonnet'),
    ],
}
