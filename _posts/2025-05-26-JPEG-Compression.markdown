---
layout: post
title:  "Compression Algorithms: JPEG"
date:   2025-05-26 01:08:51 +0000
categories: jekyll update
---

For a recent project, I've had to come up to speed on current compression algorithms, particularly video compression. Although JPEG isn't a video-compression algorithm, it's probably the one of the better starting points to learn about compression, and it's basic operations are adopted in SoTA video compression. This post is meant as a soft introduction to someone who isn't familiar with the field. 

A compression pipeline generally take some type of multimedia, runs it through an encoder, potentially sends it through a communication channel, and then recovers the media via the decoder. JPEG is a lossy compression algorithm which means that some information is lost while encoding the image into is compressed form. PNG is an example of a lossless compression algorithm, and consequently, it's compressed file sizes are generally larger compared to JPEG. Here's what the JPEG pipeline looks like:

<div class="image-container">
  <img src="/assets/img/jpeg/jpeg_pipe.png" alt="JPEG pipeline">
  <div class="image-citation">
    Source: <a href="https://www.researchgate.net/publication/363038724_A_Reliable_JPEG_Quantization_Table_Estimator" target="_blank">A Reliable JPEG Quantization Table Estimator</a>
  </div>
</div>

The JPEG compression algorithm is really just a clever way of using cosine waves and entropy coding. From a signal processing perspective, an image is a 2D signal that can be decomposed into various frequencies which is where the cosine waves come into play. This insight combined with some tips from the human visual system (HVS) lead to a timeless algorithm that maintains relevance 30+ years after ISO codified it. 

### Color Space Transformation
A color space transformation is first applied to the input image. A color space transformation is adjusting the way that an image is represented. Normally, we think of things in terms of RGB, but a lot of media applications like TV, use another color space called YCrCb. Y is the luminance and Cr/Cb stand for Chroma-red and Chroma-blue respectively. On a side note, G is highly correlated with the luma channel which aligns with HVS (we perceive green than other colors since its in the middle of the visual spectrum). This step in the pipeline just changes the image's color space into YCrCb. 

### Chroma Downsampling 
The next step in the JPEG pipeline is to perform chroma subsampling. Another interesting tidbit about the HVS is that we are much more sensitive to changes in luminance as opposed to chrominance. Therefore, we can aggressively downsample the chroma channels and still retain a high level of underlying image quality. 

There are a variety of chroma downasmpling techniques but 4:2:0 is what's used in JPEG. 
<div class="image-container">
  <img src="/assets/img/jpeg/chroma_sub.png" alt="Chroma Subsampling">
  <div class="image-citation">
    Source: <a href="https://www.rtings.com/tv/learn/chroma-subsampling" target="_blank">Chroma Subsampling</a>
  </div>
</div>
4:2:0 looks at 2x2 blocks of the chroma and replaces that 2x2 block with a single value from the upper left corner. The 4:2:0 subsampling is applied to both chroma channels. 

If we think about how much this reduces the size of the original input, the Cr and Cb blocks should be $1/4$ their original size. They make up $1/3$ of the original file size so that we've reduced the total size by $1/3 + (1/4)1/3 + (1/4)1/3 = 6/12 = 1/2$ already. It might seem like an aggressive scheme, but the difference in a normal resolution image is hardly noticeable. 

### DCT on Blocks 
Once the chroma have been downsampled the next key step is to apply the 2D - discrete cosine transform.  The idea behind the DCT is that an image can be represented as 2D signal, which in turn, can be reduced to a summation of cosine waves. 

Here's what the 2D cosine waves looks like: 
<div class="image-container">
  <img src="/assets/img/jpeg/dct-coefficients.svg" alt="Chroma Subsampling">
</div>
The color in each block represents the magnitude of the cosine waves. As you move from left to right and top to bottom, notice that the patterns become more and more complex. The waves that are used to encode the image have increasing frequency in the horizontal and vertical directions respectively. The top left portion of the image is referred to as the direct current (DC) while the other cosine waves are referred to as the alternating current (AC). 

Also notice that there are 8 rows and columns. The DCT is applied to 8x8 blocks of the image and each row + column combination represents one 2D wave to represent the 8x8 block. When the DCT is applied, this yields 8x8 = 64 total DCT coefficients. The DCT is applied to the Y, Cr, and Cb channels respectively to obtain each of their respective coefficients. 

From here on out, the remainder of operations are performed on 8x8 blocks independently. 

### Quantization 
Recall JPEG is a lossy compression algorithm. Quantization is where the loss occurs. In short, quantization is the process of representing numbers with fewer bits. The DCT coefficients are usually floating point numbers which require higher precision to represent. 

A key insight in the quantization process is high frequency components within images can be removed without a big loss in the underlying quality. High frequency components are just small scall variations within the underlying image, for example, grass or sand would comprise high frequency components within an image. These portions of the image would have greater (absolute) DCT coefficients in the bottom right corner of the image from above. 

To take advantage of this, JPEG specifies quantization tables which aggressively quantize the high frequency coefficients, pushing them to 0. For example, the DCT coefficient corresponding to the highest frequency AC would be divided by 99, while the DC would be divided by 16 in the case of the luma channel. In this way, some of the high frequency components of the image are filtered out while retaining the more important, lower frequency components. 

Here's a look at the quantization coefficients used for the luma and chroma channels: 
<div class="image-container">
  <img src="/assets/img/jpeg/quant_table.png" alt="Chroma Subsampling">
    <div class="image-citation">
    Source: <a href="https://www.researchgate.net/figure/Quantization-Matrix_fig3_279748503" target="_blank">JPEG Quantization Tables</a>
  </div>
</div>
As we mentioned, the quantization coefficients increase as you move from the low to high frequency components. Also, notice that the luma is more aggressively quantized, again, since the HVS is less sensitive to changes in chroma compared to luma. 

### Run Length Encoding + Huffman Encoding 
Once the DCT coefficients are quantized, we can use some classic techniques to compress the coefficients into an efficient bitstream representation. 

First, a zig zag scan is applied to the DCT coefficicents starting from the top left. The zig zag span helps capture more 0 runs, which in turn, helps with the compression. Longer runs of 0 will occur with zig zag scans because of the quantization tables. Once we've encoded the coefficients, they're further compressed with Huffman coding. Huffman coding deserves it's own separate explanation, but essentially, it's the most optimal, lossless classical technique to compress information into the least amount of bits possible. We'll cover Huffman coding in a separate post eventually...thanks for reading!