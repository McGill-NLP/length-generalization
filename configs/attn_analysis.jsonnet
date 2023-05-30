{
    dataset+: {
        validation_portion_skip_shuffle: true,
    },
    analyzers: [
        (import 'analyzers/attention_analyzer.jsonnet'),
    ],
}
