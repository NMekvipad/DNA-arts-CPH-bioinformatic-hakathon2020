# DNA-arts-CPH-bioinformatic-hakathon2020

We are working on DNA art challenges where we are trying to generate pieces of art from DNA SNPs information.

Our approach to this problem can be divided into 2 steps.

1. DNA encoding. For this step, we encode SNPs information from the whole human genome into a fixed-dimension vector which will be used as input to image generator as an initial seed. We take simple approach for SNPs encoding here by summing up the number of each type of bi-nucleotide SNPs found in each chromosome. Then, we standardize the counts within each SNPs among all chromosome. Thus, for each chromosome we have 16-dimension vector where each dimension corresponding to each type of SNPs. After that we concatenate 16-dimension vectors of each chromosome into one single vector representation.

2. Image generation. For image generation we use the neural network architecture called [Deep Convolutional Generative Adversarial Network (DCGAN)](https://arxiv.org/abs/1511.06434). DCGANs is a neural network architecture with 2 parts: the generator and the discriminator. The generator is a sub-network that takes fixed-length seed vector (typically random Gaussian) as and input and generates image as an output. The fake image generated from generator is then fed to discriminator. The discriminator compares the fake image with real training image and trying to tell fake image from real image. Two sub network will compete with each other during training to produce the best fake image as it can and to be the best in detecting fake image.  Our architecture is based on [this](https://www.tensorflow.org/tutorials/generative/dcgan) with twice the number of filters to enable more detail image. The input to generator is 352-dimension vector from previous step and the output is an image with of size 200 x 128 pixels (close to golden ratio for the sake of art!!!!). The artworks that we used for model training were the arts in abstract genre scraped from [WikiArts](https://www.wikiart.org/).

### Team Members
- Teodora Francesca Radut, MSc of Bioinformatics, University of Copenhagen
- Nuttapong Mekvipad, MSc of Bioinformatics, University of Copenhagen


