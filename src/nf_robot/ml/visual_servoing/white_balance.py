"""White balance for the synthetic frame ingredients: neutralize each plate, then re-light
finished frames at a random colour temperature."""

import numpy as np

# Minkowski norm for the illuminant estimate, between gray-world (1) and white-patch (inf).
SHADES_OF_GREY_P = 6
# Estimate from every Nth pixel.
ESTIMATE_PIXEL_STEP = 4

# Colour temperatures to re-light finished frames at, in Kelvin.
KELVIN_RANGE = (3000.0, 7500.0)
# Green-magenta tint, the axis fluorescent and LED lighting sits off the blackbody curve.
TINT_RANGE = 0.05
# The temperature a neutralized frame is taken to be at.
REFERENCE_KELVIN = 5000.0


def _srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x):
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.clip(x, 0, None) ** (1 / 2.4) - 0.055)


def gain_lut(gains):
    """A 256-entry uint8 lookup per channel for scaling by `gains`."""
    values = np.arange(256, dtype=np.float32) / 255.0
    linear = _srgb_to_linear(values)
    table = np.stack([_linear_to_srgb(linear * float(g)) for g in gains], axis=1)
    return np.clip(table * 255.0 + 0.5, 0, 255).astype(np.uint8)


def apply_gains(image, gains):
    """Scale an RGB or RGBA image's channels in linear light, leaving alpha alone."""
    lut = gain_lut(gains)
    out = image.copy()
    for channel in range(3):
        out[:, :, channel] = lut[:, channel][image[:, :, channel]]
    return out


def estimate_illuminant(images, alpha_min=1):
    """The illuminant these images were shot under, a shades-of-grey estimate in linear
    light normalized to green."""
    totals = np.zeros(3, dtype=np.float64)
    count = 0
    for image in images:
        pixels = image[::ESTIMATE_PIXEL_STEP, ::ESTIMATE_PIXEL_STEP]
        if pixels.shape[2] == 4:
            pixels = pixels[pixels[:, :, 3] >= alpha_min]
        pixels = pixels.reshape(-1, pixels.shape[-1])[:, :3]
        if not len(pixels):
            continue
        linear = _srgb_to_linear(pixels.astype(np.float64) / 255.0)
        totals += np.sum(linear ** SHADES_OF_GREY_P, axis=0)
        count += len(pixels)
    if not count:
        return np.ones(3)
    illuminant = (totals / count) ** (1 / SHADES_OF_GREY_P)
    return illuminant / illuminant[1]


def neutralize_gains(illuminant):
    return np.asarray(1.0 / np.asarray(illuminant, dtype=float))


def kelvin_rgb(kelvin):
    """Approximate RGB of a blackbody radiator at `kelvin`, normalised to green."""
    t = float(np.clip(kelvin, 1000.0, 40000.0)) / 100.0
    if t <= 66:
        red = 255.0
        green = 99.4708025861 * np.log(t) - 161.1195681661
    else:
        red = 329.698727446 * (t - 60) ** -0.1332047592
        green = 288.1221695283 * (t - 60) ** -0.0755148492
    if t >= 66:
        blue = 255.0
    elif t <= 19:
        blue = 0.0
    else:
        blue = 138.5177312231 * np.log(t - 10) - 305.0447927307
    rgb = np.clip(np.array([red, green, blue], dtype=np.float64), 1.0, 255.0)
    return rgb / rgb[1]


def illuminant_gains(kelvin, tint=0.0):
    """Gains that re-light a neutral image at `kelvin` with a green-magenta `tint`."""
    gains = kelvin_rgb(kelvin) / kelvin_rgb(REFERENCE_KELVIN)
    gains = gains * np.array([1.0, 1.0 + tint, 1.0])
    return gains / gains[1]


def random_illuminant_gains(rng):
    """Gains for one frame's lighting, drawn from the range live frames come back in."""
    return illuminant_gains(rng.uniform(*KELVIN_RANGE), rng.uniform(-TINT_RANGE, TINT_RANGE))
