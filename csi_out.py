import os
from threading import Thread

from matplotlib import pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm

from wifi_utility import *

# cls = 'box'
# cls = 'circle'
# cls = 'clean'
# cls = 'fall'
# cls = 'run'
cls = 'walk'
# cls = sys.argv[1]
save = True

for target_dir in ['train', 'test']:
    csi_data_root = f'data/NTU-Fi_HAR/{target_dir}_amp/{cls}/'
    out_root = f'data/csi_dataset/{target_dir}/{cls}/'
    os.makedirs(out_root, exist_ok=True)

    thread_list = []
    for idx in tqdm(range(200)):
        csi_data_path = f'{csi_data_root}{cls}{idx}.mat'

        if not os.path.exists(csi_data_path):
            continue

        csi_data = loadmat(csi_data_path)['CSIamp']
        csi_data = process_amp(csi_data)
        # print(idx, csi_data.shape)

        csi_data = 10 * np.log10(np.abs(csi_data) ** 2)
        csi_data = clean_heatmap(csi_data, 75)

        for rx_idx in range(3):
            data = csi_data[rx_idx]
            data = np.flipud(data)
            if save:
                out_path = f'{out_root}csi_{cls}_{idx}_{rx_idx}.jpg'
                # save_thread(stft_out[subcarrier], out_path)
                t = Thread(target=save_thread, args=(data, out_path))
                t.start()
                thread_list.append(t)
                # exit()

            else:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot()
                out = ax.imshow(
                    data, aspect='auto', cmap='jet'
                )
                ax.set_xlabel("Time")
                ax.set_ylabel("Subcarrier")
                ax.set_title(f"CSI Spectrogram — Stream {rx_idx}")
                fig.colorbar(out, label="Power (dB)")
                plt.show()
                exit()

        if idx % 20 == 0:
            for t in thread_list:
                t.join()

            thread_list.clear()

    for t in thread_list:
        t.join()
