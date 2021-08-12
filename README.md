

# ML Image Compression


[tensorflow compression](https://github.com/tensorflow/compression)


### Papers

- [An End-to-End Compression Framework Based on Convolutional Neural Networks](https://arxiv.org/pdf/1708.00838v1.pdf): encoder/decoder architecture tries to learn a compact, image-like representation of an image and use an image codec to store it, then uses interpolation + decoder network in the reconstruction

- [Learning Convolutional Networks for Content-weighted Image Compression](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Learning_Convolutional_Networks_CVPR_2018_paper.pdf): encoder/decoder architecture that extracts a spatial importance mask to predict optimal bit allocation ([code](https://github.com/adityassrana/Content-Weighted-Image-Compressionz))

- [High-Fidelity Generative Image Compression](https://arxiv.org/pdf/2006.09965.pdf): google paper on GANs for compression ([code](https://github.com/tensorflow/compression/tree/master/models/hific))

- [Deep Generative Models for Distribution-Preserving Lossy Compression](https://arxiv.org/pdf/1805.11057.pdf): training GANs with flexible bitrate. The resulting models behave like generative models at zero bitrate, almost perfectly reconstruct the training data at high enough bitrate ([code](https://github.com/mitscha/dplc))

- [Full Resolution Image Compression with Recurrent Neural Networks](https://openaccess.thecvf.com/content_cvpr_2017/papers/Toderici_Full_Resolution_Image_CVPR_2017_paper.pdf): google paper on using different kinds of RNNs ([blog](https://ai.googleblog.com/2016/09/image-compression-with-neural-networks.html))

- [Improved Lossy Image Compression with Priming and Spatially Adaptive Bit Rates for Recurrent Networks](https://arxiv.org/pdf/1703.10114.pdf): google paper that uses RNNs to progressively reconstruct an image

- [Nonlinear Transform Coding](https://arxiv.org/pdf/2007.03034.pdf): ([video talk](https://www.youtube.com/watch?v=x_q7cZviXkY))

- [ML Image Compression Benchmark review](https://arxiv.org/pdf/2002.03711.pdf)




