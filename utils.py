from read_data import tps, hpatches_sequence_folder, DenoiseHPatches
import csv
import cv2
import numpy as np
import os
from tqdm import tqdm


def plot_triplet(generator):
    import matplotlib.pyplot as plt
    a = next(iter(generator))
    index = np.random.randint(0, a[0]['a'].shape[0])
    plt.subplot(131)
    plt.imshow(a[0]['a'][index,:,:,0], cmap='gray') 
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.title('Anchor', fontsize=20)
    plt.subplot(132)
    plt.imshow(a[0]['p'][index,:,:,0], cmap='gray') 
    plt.title('Positive', fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.subplot(133)
    plt.imshow(a[0]['n'][index,:,:,0], cmap='gray') 
    plt.title('Negative', fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.show()

def plot_denoise(denoise_model):
    """Plots a noisy patch, denoised patch and clean patch.
    Args:
        denoise_model: keras model to predict clean patch
    """
    import matplotlib.pyplot as plt
    generator = DenoiseHPatches(['./hpatches/v_there'])
    imgs, imgs_clean = next(iter(generator))
    index = np.random.randint(0, imgs.shape[0])
    imgs_den = denoise_model.predict(imgs)
    plt.subplot(131)
    plt.imshow(imgs[index,:,:,0], cmap='gray') 
    plt.title('Noisy', fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.subplot(132)
    plt.imshow(imgs_den[index,:,:,0], cmap='gray') 
    plt.title('Denoised', fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.subplot(133)
    plt.imshow(imgs_clean[index,:,:,0], cmap='gray')
    plt.title('Clean', fontsize=20)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.show()

def generate_desc_csv(descriptor_model, seqs_test, denoise_model=None, use_clean=False, curr_desc_name='custom'):
    """Plots a noisy patch, denoised patch and clean patch.
    Args:
        descriptor_model: keras model used to generate descriptor
        denoise_model: keras model used to predict clean patch. If None,
                       will pass noisy patch directly to the descriptor model
        seqs_test: CSVs will be generated for sequences in seq_test
    """
    w = 32
    bs = 128
    output_dir = './out'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if use_clean:
        noisy_patches = 0
        denoise_model = None
    else:
        noisy_patches = 1
    for seq_path in tqdm(seqs_test):
        seq = hpatches_sequence_folder(seq_path, noise=noisy_patches)

        path = os.path.join(output_dir, os.path.join(curr_desc_name, seq.name))
        if not os.path.exists(path):
            os.makedirs(path)
        for tp in tps:
            n_patches = 0
            for i, patch in enumerate(getattr(seq, tp)):
                n_patches += 1

            patches_for_net = np.zeros((n_patches, 32, 32, 1))
            for i, patch in enumerate(getattr(seq, tp)):
                patches_for_net[i, :, :, 0] = cv2.resize(patch[0:w, 0:w], (32,32))
            ###
            outs = []
            
            n_batches = int(n_patches / bs) + 1
            for batch_idx in range(n_batches):
                st = batch_idx * bs
                if batch_idx == n_batches - 1:
                    if (batch_idx + 1) * bs > n_patches:
                        end = n_patches
                    else:
                        end = (batch_idx + 1) * bs
                else:
                    end = (batch_idx + 1) * bs
                if st >= end:
                    continue
                data_a = patches_for_net[st: end, :, :, :].astype(np.float32)
                if denoise_model:
                    data_a = np.clip(denoise_model.predict(data_a).astype(int), 0, 255).astype(np.float32)

                # compute output
                out_a = descriptor_model.predict(x=data_a)
                outs.append(out_a.reshape(-1, 128))

            res_desc = np.concatenate(outs)
            res_desc = np.reshape(res_desc, (n_patches, -1))
            out = np.reshape(res_desc, (n_patches, -1))
            np.savetxt(os.path.join(path,tp+'.csv'), out, delimiter=';', fmt='%10.5f')   # X is an array
