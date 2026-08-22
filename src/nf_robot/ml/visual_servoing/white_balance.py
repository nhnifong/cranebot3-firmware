"""White balance for the synthetic frame ingredients.

Plates are captured with the gripper camera's white balance pinned to daylight, which the
green screen needs: auto white balance drags on a green backdrop until the sheet
photographs blue, and the chroma key cannot have that. Indoors, under ~3000K room light,
pinned daylight leaves every plate yellow.

Live frames have no such cast - the control stream runs auto white balance - so the plates
are corrected here rather than at capture: the pinned preset makes the cast one constant
per run, which is exactly what makes it removable. Auto white balance at capture would
trade a known constant for a drift that changes with whatever is in shot.

Two steps, at opposite ends of the compositor. Each source is neutralized as it loads, so
floor, objects and fingers agree on what white is; then the finished frame is re-lit at a
random colour temperature, so the model sees the range of casts a real auto white balance
delivers across rooms rather than the one room the plates were shot in.
"""

import numpy as np

# Minkowski norm for the illuminant estimate. p=1 is gray-world, which takes the floor's
# own colour for a cast and over-corrects a warm carpet to pink; p=inf is white-patch,
# which rides on the brightest pixel and under-corrects. 6 is the usual compromise and is
# what looked right on the plates.
SHADES_OF_GREY_P = 6
# Estimate from every Nth pixel. The cast is a property of the lighting, so it is already
# heavily oversampled by a single frame, let alone a run of them.
ESTIMATE_PIXEL_STEP = 4

# Colour temperatures to re-light finished frames at, in Kelvin: warm room light at one
# end, overcast daylight through a window at the other.
KELVIN_RANGE = (3000.0, 7500.0)
# Green-magenta tint, the axis fluorescent and LED lighting sits off the blackbody curve.
TINT_RANGE = 0.05
# The temperature a neutralized frame is taken to already be at, so the middle of the
# range above is no change.
REFERENCE_KELVIN = 5000.0


def _srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x):
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.clip(x, 0, None) ** (1 / 2.4) - 0.055)


def gain_lut(gains):
    """A 256-entry uint8 lookup per channel for scaling by `gains`.

    A lookup rather than the arithmetic itself because the whole transform is a function
    of one byte and one gain, and the alternative is converting a few hundred cached
    plates to float and back.
    """
    values = np.arange(256, dtype=np.float32) / 255.0
    linear = _srgb_to_linear(values)
    table = np.stack([_linear_to_srgb(linear * float(g)) for g in gains], axis=1)
    return np.clip(table * 255.0 + 0.5, 0, 255).astype(np.uint8)


def apply_gains(image, gains):
    """Scale an RGB or RGBA image's channels, in linear light.

    Linear light because that is where a camera applies white balance gains; scaling the
    sRGB values directly would wash out the saturated colours the objects are made of.
    Alpha passes through untouched.
    """
    lut = gain_lut(gains)
    out = image.copy()
    for channel in range(3):
        out[:, :, channel] = lut[:, channel][image[:, :, channel]]
    return out


def estimate_illuminant(images, alpha_min=1):
    """The colour of the light these images were shot under, normalised to green.

    Shades-of-grey: the p-norm of each channel over every pixel. RGBA images contribute
    only where they are opaque, so a cutout is judged on the object rather than on the
    transparent nothing around it.

    Measured in linear light, the same space apply_gains works in. Estimating on the sRGB
    values instead would leave a residue: the gains that null a cast measured through the
    sRGB curve are not the gains that remove it.
    """
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
    """Gains that take an image shot under `illuminant` back to neutral."""
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
    """Gains that re-light a neutral image at `kelvin` with a green-magenta `tint`.

    Parameterised by temperature rather than free per-channel gains: on the blackbody
    curve red and blue move against each other, and independent jitter would spend
    training capacity on casts no room or camera can produce. photometric's per-channel
    jitter is still there for the small departures that are not the illuminant.
    """
    gains = kelvin_rgb(kelvin) / kelvin_rgb(REFERENCE_KELVIN)
    gains = gains * np.array([1.0, 1.0 + tint, 1.0])
    return gains / gains[1]


def random_illuminant_gains(rng):
    """Gains for one frame's lighting, drawn from the range live frames come back in."""
    return illuminant_gains(rng.uniform(*KELVIN_RANGE), rng.uniform(-TINT_RANGE, TINT_RANGE))
