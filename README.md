# NALU

This repository contains an custom keras layer that implements the Neural Arithmetic Logic Unit described in [arXiv:1808.00508](https://arxiv.org/abs/1808.00508).

See *basic_experiments.ipynb* for basic examples of how to use the code as well as small collection of experiments -- being expanded.

## Todo

Not in any particular order:

- [x] Working version of the code.
- [ ] Expand this Readme to include more info about the idea behind NALU.
- [ ] Replicate the experiments found in the paper -- currently done: id and +.
- [ ] Investigate the extrapolation error of NALU. 
- [ ] Add callback to the Keras layer so monitor it using it on tensorboard. 
- [ ] Following [this](https://www.reddit.com/r/MachineLearning/comments/94833t/neural_arithmetic_logic_units/e3u974x), investigate and implement any tweaks to NALU.


## References and other resources. 

Some interesting resources about NALU.

 - [arXiv:1808.00508](https://arxiv.org/abs/1808.00508) ;)
 - [This](https://www.reddit.com/r/MachineLearning/comments/94833t/neural_arithmetic_logic_units/) discussion on reddit. Includes one of the authors.
 - [This](https://medium.com/mlreview/simple-guide-to-neural-arithmetic-logic-units-nalu-explanation-intuition-and-code-64bc22605712) blog is a good introduction to NALU.

Some implementations:

 - [faizan2786/nalu_implementation](https://github.com/faizan2786/nalu_implementation): Tensorflow.
 - [bharathgs/NALU](https://github.com/bharathgs/NALU): PyTorch.
 - [kevinzakka/NALU-pytorch](https://github.com/kevinzakka/NALU-pytorch): PyTorch.
 - [kgrm/NALU](https://github.com/kgrm/NALU): Keras.
 - [titu1994/keras-neural-alu](https://github.com/titu1994/keras-neural-alu): Keras.
 - [This](http://rickyhan.com/jekyll/update/2018/08/15/neural-alu-implemented-in-python-and-assembly-x86.html) blog post presents a x86 Assembly implementation of Nalu.
 - [AtriSaxena/Neural_Arithmetic_Logic_Units-Keras](https://github.com/AtriSaxena/Neural_Arithmetic_Logic_Units-Keras): Keras.