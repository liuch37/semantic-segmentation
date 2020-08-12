# semantic_segmentation
Simple inference implementation with trained HRNet with MIT ADE20K dataset, using PyTorch 1.6.0.

## Usage:
1. Download pretrained model from http://sceneparsing.csail.mit.edu/model/pytorch and store them in './ade20k-hrnetv2-c1'

2. Specify your test input image directory and test output image directory in semantic_segmentation.py

image_path

output_path

3. Specify if you want to use GPU or not (-1 for cpu, others for gpu index) in semantic_segmentation.py

gpu = -1

4. Run python semantic_segmentation.py

5. All the inference results will be stored in your output_path

## source
Original code: https://github.com/CSAILVision/semantic-segmentation-pytorch.

HRNet: https://arxiv.org/abs/1904.04514.
