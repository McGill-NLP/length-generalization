local jsonlines = (import 'jsonlines.jsonnet');

jsonlines + {
    dataset+: {
        data_root: 'data',
        name: 'glue_mrpc',
        split: std.extVar("APP_DS_SPLIT"),
        label_list: ['not_equivalent', 'equivalent'],
        input_prompt: 'sentence1|sentence2',
        is_regression: false,
    },
}
