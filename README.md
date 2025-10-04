# REPORT - Caroline Grand-Clement

## Part 1:

It was a challenge to understand how to work with the data we had, which information we needed and which we could discard. I used the adjusted bounded boxes of each annotation to create a binary mask of where the characters are in the image. 
To save space, I first resized the images and then calculated the coordinates of the bounded boxes proportionally. I parsed the coordinates to integers to be able to use them more easily as indexes to create my mask using a matrix. The resizing and parsing brings a lot of imprecision, and may be to blame for the poor results discussed later on. However, I was often running into out-of-memory CUDA errors, so this is the solution I opted for. 


## Part 2:
    
I decided to split my training, test, and development sets in a 8/1/1 fashion. When first developing the models, I ran them with the development set with 3 epochs, so as to minimize computation while I figured out if the models ran appropriately. 

The first model I created is an extremely simple model in regards to the task. Still, I wanted to test out how a model with only a few layers (essentially only convolutional, pooling, and upsampling) would do. The results were, as expected, extremely poor. (Jaccard Index = 0.0045)
For the second model, I took inspiration from the UNet model, with a reduced number of layers because of memory space. I am only using two "steps" in the encoder and decoder blocks, using convolutional layers and pooling or upsampling layers in each. I had hoped that this model would perform much better than the first one, and to my surprise, it seems equally as bad. 
I trained both models on the training set for 50 epochs each, with batch size 24. However, the loss does not seem to indicate much learning for either.
For both models, I use binary cross entropy loss.


## Part 3: 
For testing, I ran the test dataset through the model and then used the binary Jaccard index.
As mentioned before, the models perform very badly. For the first model, this is explained easily by the fact that it does not have many layers, and the upsampling is very simple, which introduces a lot of imprecision in the output. Its Jaccard index was 0.0006.
For the UNet-inspired model, I was disappointed to see it did not learn well. Its Jaccard index was 0.0016.

To visualize the golden mask and predicted outputs, I plotted the three masks I had side by side. It then became clear why the UNet model had not been efficient. Because of the resizing, the target masks were almost exclusively 0s, indicating that nothing should be identified. Figure 1 shows this well, with the first plot being the original image resized to 128, the second is the target mask, the third is the output of the simple model, and the fourth is the output of the UNet model. We can see that the latter has in fact learned information about the image quite well, only not how to recognize where characters are (because the target does not tell it there are any characters).

I then decided to resize the target masks after computing them in 2048 * 2048 size, and then update the values to 0 or 1 as they had been 'resized' to floats. 

This improved the models consideradly (Jaccard index increased by 100 times), but there is still a lot of imprecision.
The obtained Jaccard indexes after this improvement were 0.1378 for the simple model, and 0.1648 for the UNet model.


### Figure 1: 

[Figure showing golden mask as only 0s](resize_before_mask.png "Figure 1")

### Figure 2: 

[Figure showing golden mask after resizing](resize_after_mask.png "Figure 2")
