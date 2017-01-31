# FastVideoStyleTransfer
**This is Python 3 rewrite of "Fast Style Transfer in [TensorFlow]"(https://github.com/tensorflow/tensorflow)**

On Windows Tensorflow no longer supports Python 2, so in order to make the project run on Windows, a number of python files were rewritten in Python 3 syntax.

This project allows you to transfer the style of an image or a video sequence based on the style image provided.

The implementation is based off of a combination of Gatys' A Neural Algorithm of Artistic Style, Johnson's Perceptual Losses for Real-Time Style Transfer and Super-Resolution, and Ulyanov's Instance Normalization.

## Video Stylization ##

Here we transformed every frame in a video, then combined the results. 

## Image Stylization ##

We added styles from various paintings to a photo of Chicago. Click on thumbnails to see full applied style images.    
   
# Implementation Details #

Our implementation uses TensorFlow to train a fast style transfer network. We use roughly the same transformation network as described in Johnson, except that batch normalization is replaced with Ulyanov's instance normalization, and the scaling/offset of the output tanh layer is slightly different. We use a loss function close to the one described in Gatys, using VGG19 instead of VGG16 and typically using "shallower" layers than in Johnson's implementation (e.g. we use relu1_1 rather than relu1_2). Empirically, this results in larger scale style features in transformations.

# Documentation #

Training Style Transfer Networks

Use style.py to train a new style transfer network. Run python style.py to view all the possible parameters. Training takes 4-6 hours on a Maxwell Titan X. More detailed documentation here. Before you run this, you should run setup.sh. Example usage:

python style.py --style path/to/style/img.jpg \
  --checkpoint-dir checkpoint/path \
  --test path/to/test/img.jpg \
  --test-dir path/to/test/dir \
  --content-weight 1.5e1 \
  --checkpoint-iterations 1000 \
  --batch-size 20
Evaluating Style Transfer Networks

Use evaluate.py to evaluate a style transfer network. Run python evaluate.py to view all the possible parameters. Evaluation takes 100 ms per frame (when batch size is 1) on a Maxwell Titan X. More detailed documentation here. Takes several seconds per frame on a CPU. Models for evaluation are located here. Example usage:

python evaluate.py --checkpoint path/to/style/model.ckpt \
  --in-path dir/of/test/imgs/ \
  --out-path dir/for/results/
Stylizing Video

Use transform_video.py to transfer style into a video. Run python transform_video.py to view all the possible parameters. Requires ffmpeg. More detailed documentation here. Example usage:

python transform_video.py --in-path path/to/input/vid.mp4 \
  --checkpoint path/to/style/model.ckpt \
  --out-path out/video.mp4 \
  --device /gpu:0 \
  --batch-size 4

# Result #

## View Video ##

**Input Video**

[https://github.com/trendmaster1/FastVideoStyleTransfer/tree/master/examples/results/view.mp4](https://github.com/trendmaster1/FastVideoStyleTransfer/tree/master/examples/results/view.mp4)

**Output Video**

[https://github.com/trendmaster1/FastVideoStyleTransfer/tree/master/examples/results/view_out.mp4](https://github.com/trendmaster1/FastVideoStyleTransfer/tree/master/examples/results/view_out.mp4)

## Pool Video ##

**Input Video**

[https://github.com/trendmaster1/FastVideoStyleTransfer/tree/master/examples/results/pool.mp4](https://github.com/trendmaster1/FastVideoStyleTransfer/tree/master/examples/results/pool.mp4)

**Output Video**

[https://github.com/trendmaster1/FastVideoStyleTransfer/tree/master/examples/results/pool_output.mp4](https://github.com/trendmaster1/FastVideoStyleTransfer/tree/master/examples/results/pool_out.mp4)
