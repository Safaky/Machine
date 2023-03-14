

# Classification of X-ray Images of the Lungs
Using machine learning and Deep learning, the ability to group x-ray images of the lungs with healthy - unhealthy and multiple classification (grouping according to various lung diseases) has emerged as a very useful and time-saving way for medical radiologists and physicians.
The basic steps required for the development of a suitable deep learning and machine learning algorithm are explained.

* Various segmentation paths and deep learning algorithms are available for the analysis of lung images. The important steps to take before using them are described below.



## Image-Preprocessing

In medical image analysis, images must first be preprocessed. This process is very important in terms of the accuracy of the results to be obtained from the analysis. Noise removal, thresholding, blurring, resizing and histogram equalization are the basic image preprocessing methods.

If we do not have enough data, a small number of data can be reproduced by diversifying by using the method known as data duplication method. This diversification includes operations such as rotating the image, mirroring, changing the color channel, and cropping.

Examples of operations such as resizing, changing the image format, cropping, rotating, mirroring, rgb and gray scale conversion, splitting and merge, HOG histogram application and masking are given.


## Data Labeling
Data labeling is essential for the initial phase of the model, where it learns from humans. At this stage, the model learns what a diseased and disease-free lung looks like. Multi-labeling can also be done for multi-classification, such as binary tagging. Learning by labeling data is called supervised learning, and models built without it are called unsupervised learning. 
* Supervised learning is the most common and applies to classification and segmentation processes.
