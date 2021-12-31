In this code, we want to cluster a bunch of unlabeled faces into clusters. So, there exist some issues:
1. Firstly, we should use which of the algorithms for clustering?
2. Secondly, how can we show a face for the algorithm?
3. Lastly, how measure the similarity of faces?
- For the first issue, I think that it is better to use an algorithm that doesn't need the number of classes as input (like K-Means). Consequently, I prefer to use an algorithm that just needs a threshold for similar faces, and I used the [Mean-shift](https://ieeexplore.ieee.org/document/1000236) algorithm for this purpose. Mean-Shift algorithm clusters the data by **shifting** the mean of the data towards the densest space, and at the final stage we have multi-points (center of clusters). Moreover, in the test phase, the input data (mostly vector), is compared to the centers and it will belong to the cluster that is more near to its center. I liked to use the implemented [Mean-shift code](https://pythonprogramming.net/mean-shift-from-scratch-python-machine-learning-tutorial/)from scratch by [Harrison](https://github.com/Sentdex). 
- In order to demonstrate a face in the feature vector, a trained network like mobile-net that embeds a face into a 1D feature vector can be used. In this code [facenet-pytorch](https://github.com/timesler/facenet-pytorch) was utilized for face embedding. It converts the input image face to a 512 dimensions vector.
- Finally, for comparing two faces, the cosine-similarity was used, and two faces whose cosine-similarity is more than 0.8 (or for verification is less than 0.2) belong to the same classes.

#### <span style="color:red"> Attention </span>
The cosine value of the angle between two vectors is always between [-1,1], but, in this project, the value was scaled between [0,1] for normalization:
<span style="color:orange;">Word up</span>
![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}](https://latex.codecogs.com/svg.latex?\Large&space;x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}) 
