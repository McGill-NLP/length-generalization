local base = (import 'base.jsonnet');

base + {
    method: 'grid',
    metric: {
        goal: 'maximize',
        name: 'pred/valid_acc_overall',
    },
    parameters: {},
}
