"""Versioned, reproducible detector composites; measurements use source planes."""
import json
import os
import re

import numpy as np

MAPPING_FILENAME = 'channel_color_mapping.txt'


def validate_mapping(mapping):
    """Return a canonical mapping. Channel indices are zero based."""
    if not isinstance(mapping, dict) or type(mapping.get('version')) is not int or mapping['version'] != 1:
        raise ValueError('Channel-color mapping must have version 1.')
    if mapping.get('normalization') != 'per_tile_minmax':
        raise ValueError('Unsupported channel normalization; expected per_tile_minmax.')
    entries = mapping.get('channels')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Select at least one detection channel.')
    channels, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Each channel needs an index and an RGB hex color.')
        index, color = entry.get('index'), entry.get('color')
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError('Channel indices must be unique nonnegative integers (first channel: 0).')
        if not isinstance(color, str) or not re.fullmatch(r'#[0-9a-fA-F]{6}', color):
            raise ValueError('Colors must be RGB hex values, for example #0000FF.')
        if color.lower() == '#000000':
            raise ValueError('Black hides the channel; choose a non-black color.')
        seen.add(index)
        channels.append({'index': index, 'color': color.upper()})
    return {'version': 1, 'normalization': 'per_tile_minmax',
            'channels': sorted(channels, key=lambda entry: entry['index'])}


def load_mapping(path):
    with open(path, encoding='utf-8') as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError('Channel-color mapping must be a JSON object.')
    # Accept either the extraction sidecar or a trained detector's parameters.
    return validate_mapping(data.get('channel_color_mapping', data))


def save_mapping(path, mapping):
    mapping = validate_mapping(mapping)
    if os.path.exists(path) and load_mapping(path) != mapping:
        raise ValueError('This output folder has a different channel-color mapping. Choose a new folder.')
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(mapping, stream, indent=2)
        stream.write('\n')


def read_channels(path):
    """Read CYX planes (first LIF image, first time point and Z plane).

    Use TIFF axes rather than guessing from dimension sizes. Generic page
    stacks (QYX/IYX) are treated as channels; non-singleton unknown axes fail.
    """
    if os.path.splitext(path)[1].lower() == '.lif':
        from readlif.reader import LifFile
        image = next(LifFile(path).get_iter_image())
        return np.stack([np.asarray(plane) for plane in image.get_iter_c(t=0, z=0)])
    from tifffile import TiffFile
    with TiffFile(path) as image:
        series = image.series[0]
        array, axes = series.asarray(), series.axes
    for axis in ('T', 'Z'):
        if axis in axes:
            position = axes.index(axis)
            array = np.take(array, 0, axis=position)
            axes = axes[:position] + axes[position + 1:]
    for position in range(len(axes) - 1, -1, -1):
        if axes[position] not in 'YX' and array.shape[position] == 1:
            array = np.take(array, 0, axis=position)
            axes = axes[:position] + axes[position + 1:]
    if axes == 'YX':
        return array[None, ...]
    if len(axes) == 3 and 'Y' in axes and 'X' in axes:
        channel_axis = next(i for i, axis in enumerate(axes) if axis not in 'YX')
        if axes[channel_axis] in 'CSQI':
            return np.transpose(array, (channel_axis, axes.index('Y'), axes.index('X')))
    raise ValueError('Unsupported image axes %s; provide a single channel stack with Y/X dimensions.' % axes)


def validate_channel_indices(channels, indices):
    if channels.ndim != 3:
        raise ValueError('Expected a CYX channel stack.')
    for index in indices:
        if type(index) is not int or not 0 <= index < channels.shape[0]:
            raise ValueError('Channel %s is unavailable; image has %s channels (starting at 0).' % (index, channels.shape[0]))


def compose_rgb(channels, mapping):
    """Normalize each unpadded tile plane, add its RGB contribution, clip."""
    mapping = validate_mapping(mapping)
    validate_channel_indices(channels, [entry['index'] for entry in mapping['channels']])
    rgb = np.zeros((*channels.shape[1:], 3), dtype=np.float64)
    for entry in mapping['channels']:
        plane = np.asarray(channels[entry['index']], dtype=np.float64)
        if not np.isfinite(plane).all():
            raise ValueError('Detection channels contain non-finite intensities.')
        low, high = plane.min(), plane.max()
        # Constant tiles have no contrast; do not turn uniform background white.
        normalized = (plane - low) / (high - low) if high > low else np.zeros_like(plane)
        color = np.array([int(entry['color'][i:i+2], 16) for i in (1, 3, 5)])
        rgb += normalized[..., None] * color
    return np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)


def detection_tile(channels, mapping, size, black_background=True, bgr=False):
    """Shared extraction/inference conversion, including edge padding."""
    if type(size) is not int or size <= 0:
        raise ValueError('Tile size must be a positive integer.')
    rgb = compose_rgb(channels, mapping)
    height, width = rgb.shape[:2]
    if height > size or width > size:
        raise ValueError('Input tile exceeds the detector frame size.')
    output = np.full((size, size, 3), 0 if black_background else 255, dtype=np.uint8)
    output[:height, :width] = rgb
    return output[..., ::-1].copy() if bgr else output


def extract_composites(path, output_folder, size, mapping, black_background=True):
    from tifffile import imwrite
    mapping = validate_mapping(mapping)
    channels = read_channels(path)
    validate_channel_indices(channels, [entry['index'] for entry in mapping['channels']])
    if type(size) is not int or size <= 0:
        raise ValueError('Tile size must be a positive integer.')
    os.makedirs(output_folder, exist_ok=True)
    save_mapping(os.path.join(output_folder, MAPPING_FILENAME), mapping)
    stem = os.path.splitext(os.path.basename(path))[0]
    for y in range(0, channels.shape[1], size):
        for x in range(0, channels.shape[2], size):
            rgb = detection_tile(channels[:, y:y+size, x:x+size], mapping, size, black_background)
            # Lossless TIFF avoids JPEG color changes between training/inference.
            imwrite(os.path.join(output_folder, '%s_x%d_y%d_rgb.tif' % (stem, x, y)), rgb, photometric='rgb')


def training_mapping(folder, mapping=None):
    """Load an extraction sidecar automatically, reject conflicting metadata."""
    if mapping is not None:
        mapping = validate_mapping(mapping)
    sidecar = os.path.join(folder, MAPPING_FILENAME)
    if os.path.isfile(sidecar):
        saved = load_mapping(sidecar)
        if mapping is not None and saved != mapping:
            raise ValueError('Training mapping differs from channel_color_mapping.txt in the image folder.')
        return saved
    return mapping
