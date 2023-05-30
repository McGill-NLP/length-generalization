local base = (import 'base.jsonnet');

base + {
    tokenizer+: {
        type: 'whitespace',
    },
}
