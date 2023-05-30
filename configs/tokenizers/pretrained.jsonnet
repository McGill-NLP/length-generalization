local base = (import 'base.jsonnet');

base + {
    tokenizer+: {
        type: 'pretrained',
        hf_model_name: $.model.hf_model_name,
    },
}
