# Semantic Segmentation
Simple inference implementation with trained HRNet on MIT ADE20K dataset, using PyTorch 1.6.0. Most of the code taken from [1]. Purpose of this project is to unify sky pixels with ultra high prediction confidence to a single color, in order to remove sun light effect and inconsistent cloud texture.

## Usage
1. Download pretrained model from http://sceneparsing.csail.mit.edu/model/pytorch and store them in './ade20k-hrnetv2-c1/'

2. Specify your test input image directory and test output image directory in semantic_segmentation.py
```
   image_path = './input/'

   output_path = './output/'
```
3. Specify if you want to use GPU or not (-1 for cpu, others for gpu index) in semantic_segmentation.py
```
   gpu = -1
```
4. Run ```python semantic_segmentation.py```

5. All the inference results will be stored in your output_path

## Results

### semantic map

![Image of semantic map](https://github.com/liuch37/semantic-segmentation/blob/master/misc/ADE_test_00000272.png)

### sky filtered image

![Image of semantic map](https://github.com/liuch37/semantic-segmentation/blob/master/misc/ADE_test_00000272_filtered.png)

## Source
[1] Original code: https://github.com/CSAILVision/semantic-segmentation-pytorch.

[2] HRNet: https://arxiv.org/abs/1904.04514.
