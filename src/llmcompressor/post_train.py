


def _preprocess(...):
    ...
    
    
def _post_process(...):
    ...

def post_train(
    data,
    dataset_args,
    dataloader_args,
    dataset_processor,
    dataset_sampler,
    dataset_collator,
    model,
    processor,
    recipe,
    recipe_args,
    recipe_stage,
):
    model = check_load_model(model)
    processor = check_load_processor(processor, model)
    loader = check_load_dataloader(data, dataset_args, dataloader_args, dataset_processor if dataset_processor else processor, dataset_sampler, dataset_collator)

    reset_session()  # argument could be made to make this create_session and add a context to encapsulate everything
    initialize(recipe, recipe_stage, recipe_args, model)

    for batch in loader:
        callbacks.batch_start()
        model(batch)
        callbacks.batch_end()

    finalize()

    return model