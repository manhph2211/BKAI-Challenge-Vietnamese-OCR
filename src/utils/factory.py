from .register import is_model, model_entrypoints


def create_model(
    model_name,
    pretrained=None,
    checkpoint_path='',
    **kwargs
):
    if is_model(model_name):
        create_fn = model_entrypoints(model_name)
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

    model = create_fn(pretrained=pretrained, **kwargs)

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    return model