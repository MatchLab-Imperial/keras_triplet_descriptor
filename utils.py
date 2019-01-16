from read_data import tps, hpatches_sequence_folder, DenoiseHPatches
import csv
import cv2
import numpy as np
import os


def plot_denoise(denoise_model):
    """Plots a noisy patch, denoised patch and clean patch.
    Args:
        denoise_model: keras model to predict clean patch
    """
    import matplotlib.pyplot as plt
    generator = DenoiseHPatches(['./hpatches-release/i_ajuntament'])
    imgs, imgs_clean = next(iter(generator))
    index = np.random.randint(0, imgs.shape[0])
    imgs_den = denoise_model.predict(imgs)
    plt.subplot(131)
    plt.imshow(imgs[index,0], cmap='gray') 
    plt.subplot(132)
    plt.imshow(imgs_den[index,0], cmap='gray') 
    plt.subplot(133)
    plt.imshow(imgs_clean[index,0], cmap='gray')
    plt.show()

def generate_desc_csv(descriptor_model, denoise_model, seqs_test):
    """Plots a noisy patch, denoised patch and clean patch.
    Args:
        descriptor_model: keras model used to generate descriptor
        denoise_model: keras model used to predict clean patch. If None,
                       will pass noisy patch directly to the descriptor model
        seqs_test: CSVs will be generated for sequences in seq_test
    """
    w = 65
    bs = 128
    output_dir  = './out'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for seq_path in seqs_test:
        curr_desc_name = 'res'
        seq = hpatches_sequence_folder(seq_path, noise = 1)

        path = os.path.join(output_dir, os.path.join(curr_desc_name, seq.name))
        if not os.path.exists(path):
            os.makedirs(path)
        for tp in tps:
            print(seq.name+'/'+tp)
            if os.path.isfile(os.path.join(path, tp+'.csv')):
                continue
            n_patches = 0
            for i,patch in enumerate(getattr(seq, tp)):
                n_patches+=1

            patches_for_net = np.zeros((n_patches, 1, 64, 64))
            uuu = 0
            for i,patch in enumerate(getattr(seq, tp)):            
                patches_for_net[i,0,:,:] = cv2.resize(patch[0:w,0:w],(64,64))
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

                data_r = np.zeros((data_a.shape[0], 1, 32, 32))
                for i in range(data_a.shape[0]):
                    data_r[i, 0] = cv2.resize(data_a[i, 0], (32, 32))
                # compute output
                
                out_a = descriptor_model.predict(x=data_r)
                outs.append(out_a.reshape(-1, 128))

            res_desc = np.concatenate(outs)
            print(res_desc.shape, n_patches)
            res_desc = np.reshape(res_desc, (n_patches, -1))
            out = np.reshape(res_desc, (n_patches,-1))
            np.savetxt(os.path.join(path,tp+'.csv'), out, delimiter=';', fmt='%10.5f')   # X is an array
