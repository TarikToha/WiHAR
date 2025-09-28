import os
from threading import Thread

from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.signal import stft
from tqdm import tqdm

from wifi_utility import *

save = True
cls = 'circle'
# cls = sys.argv[1]
fs = 125
nperseg = 8

for target_dir in ['train', 'test']:
    csi_data_root = f'data/NTU-Fi_HAR/{target_dir}_amp/{cls}/'
    out_root = f'data/stft_dataset/{target_dir}/{cls}/'
    os.makedirs(out_root, exist_ok=True)

    for idx in tqdm(range(200)):
        csi_data_path = f'{csi_data_root}{cls}{idx}.mat'

        if not os.path.exists(csi_data_path):
            continue

        csi_data = loadmat(csi_data_path)['CSIamp']
        csi_data = process_amp(csi_data)
        # print(idx, data.shape)

        stft_out = []
        for stream in csi_data:
            stft_stream = []
            for subcarrier in stream:
                f, t, Zxx = stft(
                    subcarrier, fs=fs, nperseg=nperseg, noverlap=nperseg // 2
                )
                stft_stream.append(Zxx)

            stft_stream = np.array(stft_stream)  # shape: (subcarriers, freq_bins, time_bins)
            stft_out.append(stft_stream)

        stft_out = np.array(stft_out)  # shape: (streams, subcarriers, freq_bins, time_bins)
        stft_out = np.abs(stft_out) ** 2

        thread_list = []
        for subcarrier in range(114):
            csi_data = stft_out[:, subcarrier]
            csi_data = np.vstack((csi_data[0], csi_data[1], csi_data[2]))
            if save:
                out_path = f'{out_root}stft_{cls}_{idx}_{subcarrier}.jpg'
                # save_thread(stft_out[subcarrier], out_path)
                t = Thread(target=save_thread, args=(csi_data, out_path))
                t.start()
                thread_list.append(t)
                # exit()

            else:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot()
                out = ax.imshow(
                    csi_data, aspect='auto', cmap='jet', origin='lower'
                )
                ax.set_xlabel("Time Bins")
                ax.set_ylabel("Frequency Bins")
                ax.set_title(f"STFT Spectrogram — Subcarrier {subcarrier}")
                fig.colorbar(out, label="Power (W)")
                plt.show()
                exit()

        for t in thread_list:
            t.join()
