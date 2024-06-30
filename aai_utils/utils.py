import predictive_coding as pc
import torch


def create_model(
    predictive_coding,
    acf,
    model_type_order,
    cnn_layers,
    linear_layers,
    loss_fn='',
    pt_model_path=None,
    trainable_layers=None,
):

    model_type_order = eval(model_type_order)

    model = []

    for cnn_key, cnn_layer in cnn_layers.items():
        for model_type in model_type_order:
            if model_type == 'Weights':
                model_ = eval(cnn_layer['fn'])(
                    **cnn_layer['kwargs']
                )
            elif model_type == 'Acf':
                model_ = eval(acf)()
            elif model_type == 'PCLayer':
                model_ = pc.PCLayer()
            elif model_type == 'MaxPool':
                model_ = torch.nn.MaxPool2d(
                    kernel_size=2,
                    stride=2,
                )
            elif model_type == 'Dropout2d':
                model_ = torch.nn.Dropout2d()
            elif model_type == 'Dropout':
                continue
            elif model_type == 'BatchNorm':
                model_ = torch.nn.BatchNorm2d(cnn_layer['kwargs']['out_channels'])
            else:
                raise ValueError('model_type not found')

            model.append(model_)

    model.append(torch.nn.Flatten())

    for linear_key, linear_layer in linear_layers.items():
        if linear_key == 'last':
            model_ = eval(linear_layer['fn'])(
                **linear_layer['kwargs']
            )
            model.append(model_)

            in_features_last = linear_layer['kwargs']['in_features']
        else:
            for model_type in model_type_order:
                if model_type == 'Weights':
                    model_ = eval(linear_layer['fn'])(
                        **linear_layer['kwargs']
                    )
                elif model_type == 'Acf':
                    model_ = eval(acf)()
                elif model_type == 'PCLayer':
                    model_ = pc.PCLayer()
                elif model_type == 'Dropout':
                    model_ = torch.nn.Dropout()
                elif model_type == 'BatchNorm':
                    model_ = torch.nn.BatchNorm1d(linear_layer['kwargs']['out_features'])
                else:
                    continue

                model.append(model_)

    if loss_fn == 'cross_entropy':
        model.append(torch.nn.Softmax())

    # decide pc_layer
    for model_ in model:
        if isinstance(model_, pc.PCLayer):
            if not predictive_coding:
                model.remove(model_)

    # # initialize
    # for model_ in model:
    #     if isinstance(model_, torch.nn.Linear):
    #         eval(init_fn)(
    #             model_.weight,
    #             **init_fn_kwarg,
    #         )

    # create sequential
    model = torch.nn.Sequential(*model)

    print(list(model.parameters()))
    if pt_model_path:
        model.load_state_dict(torch.load(pt_model_path))
        model[-1] = torch.nn.Linear(in_features_last, 2, bias=True)

        for param in list(model.parameters())[:-1 * trainable_layers]:
            param.requires_grad = False

    print("MODEL ARCHITECTURE:\n")
    print(model)

    return model
