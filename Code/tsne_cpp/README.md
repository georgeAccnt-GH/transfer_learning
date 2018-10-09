# TSNE CPP

Repo with TSNE implementation in cpp with CUDA: https://github.com/georgedimitriadis/t_sne_bhcuda

In [Benchmark.ipynb](tsne_cpp/t_sne_bhcuda/Benchmark.ipynb) there is a benchmark comparing TSNE CUDA implementation and TSNE sklearn.

With a 1000x1000 data matrix, TSNE CUDA takes 20.6s while TSNE sklearn takes 32.3s. The ratio sklearn/cuda is 1.56. However, it is important to note that the plots are different, maybe there are differences in the implementations of sklearn and cuda.

