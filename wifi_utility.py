import cv2
import numpy as np
from PIL import Image
from scipy.fft import ifft


def process_amp(x: np.ndarray) -> np.ndarray:
    """Normalize and reshape CSI amplitude data."""
    # Normalize (dataset-specific values)
    x = (x - 42.3199) / 4.9802

    # Downsample: 2000 → 500 (by taking every 4th sample)
    x = x[:, ::4]

    # Reshape to (streams, subcarriers, time)
    x = x.reshape(3, 114, -1)
    return x


def save_thread(image_data, out_file):
    """Save the image data to disk."""
    image_data = np.flipud(image_data)
    image_data = cv2.normalize(
        image_data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    image_data = cv2.resize(image_data, (1386, 1386), interpolation=cv2.INTER_CUBIC)
    image_data = cv2.applyColorMap(image_data, cv2.COLORMAP_JET)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

    image_data = Image.fromarray(image_data)
    # image_data = image_data.resize(size=(1386, 1386), resample=Image.Resampling.LANCZOS)
    image_data.save(out_file, dpi=(300, 300))


def get_cir_ifft(csi_data, n=None):
    cir_ifft = csi_data - csi_data.mean(axis=1, keepdims=True)
    cir_ifft *= np.hanning(cir_ifft.shape[1]).reshape(-1, 1)
    cir_ifft = ifft(cir_ifft, n=n, axis=1)
    return cir_ifft


def clean_heatmap(fft_in, q):
    data = fft_in.copy()
    vmin = np.percentile(data, q)
    data[data < vmin] = vmin
    return data
