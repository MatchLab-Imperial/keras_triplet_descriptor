# N-HPatches Baseline Code

This repository contains the baseline code for the Deep Learning Coursework Project (EE3-25).
 
The project aims to create a learned descriptor that is able to perform matching, verification and retrieval tasks successfully on N-HPatches. N-HPatches is a noisy version of [HPatches](https://github.com/hpatches) dataset.

We will keep updating the repository, with more detailed explanations and improved code. Please, be aware of the version you are using and keep track of future changes. 
<br />

## HPatches Dataset

HPatches is based on 116 sequences, 57 sequences presenting photometric changes, while the other 59 sequences show geometric deformations due to viewpoint change. A sequence includes a reference image and 5 target images each one with varying photometric or geometric changes. Homographies relating reference images with target images are provided.

Patches are sampled in the reference image using a combination of local feature extractors (Hessian, Harris and DoG detector). The patch orientation is estimated using a single major orientation using Lowe's method. No affine adaptation is used, therefore all patches are square regions in the reference image. Afterward, patches are projected on the target images using the ground-truth homographies. Hence, a set of corresponding patches contains one patch from each image in the sequence.

In practice, when a detector extracts corresponding regions in different images, it does so with a certain amount of noise. In order to simulate this noise, detections are perturbed using three settings: EASY, HARD and TOUGH, each one created by increasing the geometric transformation applied, resulting in increased detector noise. In other words, as bigger is the geometric transformation, harder will be to solve the task at hand. 

Following images show the reprojected easy/hard patches in the target images together with the extracted patches, which can be found in the original [github repository](https://github.com/hpatches/hpatches-dataset):

<p align="center">
 <img src=./Images/images_easy.png> <br/>
 <i>Image 1: Visualization of the easy patches locations in the target images.</i>
</p>

<p align="center">
 <img src=./Images/patches_easy.png> <br/>
 <i>Image 2: Extracted easy patches from the example sequence.</i>
</p>


<p align="center">
 <img src=./Images/images_hard.png> <br/>
 <i>Image 3: Visualization of the hard patches locations in the target images.</i>
</p>

<p align="center">
 <img src=./Images/patches_hard.png> <br/>
 <i>Image 4: Extracted hard patches from the example sequence.</i>
</p>

You can find more details on original HPatches dataset [here](https://arxiv.org/pdf/1704.05939.pdf).

## N-HPatches Dataset

N-HPatches dataset is a noisy version of HPatches. Each sequence, in addition to original patches, contains a noisy version for each one of them. Different noises are added depending on the three settings described above: EASY, HARD and TOUGH, from low to high noises. 

Patches are downsampled from original size, 65x65, to 32x32. We will test performance of the descriptor in noisy patches, however, clean patches can be used in the training stage.

**N-HPatches Dataset** can be found [here](https://imperialcollegelondon.box.com/shared/static/ah40eq7cxpwq4a6l4f62efzdyt8rm3ha.zip). Data structure is the same as in HPatches, therefore, please refer to original [paper](https://arxiv.org/pdf/1704.05939.pdf) for extra details.
<br />

## Baseline approach

This repository contains a baseline pipeline for the N-HPatches descriptor project. 

The pipeline followed is based on two consecutive networks. The first network is in charge of getting a cleaner version of the noisy input patch, while the next network will get the final descriptor. 

The architectures that have been chosen as a first approach is a shallow [UNet](https://arxiv.org/pdf/1505.04597.pdf) for the denoising part, and [L2-Net](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8100132) architecture for the descriptor. 

In order to train the denoising model, the baseline code uses the Mean Absolute Error (MAE) as a loss function between the output of the network for a noisy patch and its corresponding cleaned patch. 

On the other hand, baseline code trains the descriptor based on the [Triplet loss](http://www.bmva.org/bmvc/2016/papers/paper119/paper119.pdf), which takes an anchor patch, a negative patch and a positive patch. The idea is to train the network so the descriptors from the anchor and positive patch have a low distance between them, and the negative and anchor patch has a large distance between them. In order to do so, the code generates three instances of the network (sharing the weights) and the training triplets. Further architectures or loss functions could be used in order to improve both steps separately, or even, merge them and optimize them together.

<p align="center">
 <img src=./Images/Inference_time.png> <br/>
 <i>Image 5: Pipeline in inference time.</i>
</p>

<br />

## Evaluation metrics

We use the mean average precision (mAP) on three different tasks: Patch Verification, Image Matching and Patch Retrieval, to evaluate the descriptors. Those tasks have been designed to imitate typical use cases of local descriptors. The final score is the mean of the mAPs on all tasks. To learn more details on how the evaluation is computed, refer to the [HPatches benchmark](https://github.com/hpatches/hpatches-benchmark).

**Patch Verification** measures the ability of a descriptor to classify whether two patches are extracted from the same measurement.

**Image Matching**  tests to what extent a descriptor can correctly identify correspondences in two images. To evaluate the image matching, 

**Patch Retrieval** tests how well a descriptor can match a query patch to a pool of patches extracted from many images, including many distractors.

