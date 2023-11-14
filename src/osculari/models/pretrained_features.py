"""
Extracting features from different layers of a pretrained model.
"""


def resnet_slice(layer, is_clip=False):
    if layer == 'area0':
        layer = 10 if is_clip else 4
    elif layer == 'area1':
        layer = 11 if is_clip else 5
    elif layer == 'area2':
        layer = 12 if is_clip else 6
    elif layer == 'area3':
        layer = 13 if is_clip else 7
    elif layer == 'area4':
        layer = 14 if is_clip else 8
    elif layer in ['encoder', 'fc']:
        layer = None
    else:
        RuntimeError('Unsupported layer %s' % layer)
    return layer


def resnet_layer(layer, is_clip=False):
    slice_ind = resnet_slice(layer, is_clip=is_clip)
    layer_ind = -1 if slice_ind is None else slice_ind - 1
    return layer_ind
