{
    dataset+: {
        validation_portion_skip_shuffle: true,
    },
    analyzers: [
        {
            type: 'attention_kl',
            no_pe_run_ids: std.parseJson(std.extVar('APP_NO_PE_RUN_IDS')),
            seed: std.parseInt(std.extVar('APP_SEED')),
            attention_analysis_root_dir: std.extVar('APP_ATTENTION_ANALYSIS_ROOT_DIR'),
        },
    ],
}
