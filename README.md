repo to create face puppets, based on [fastai](fast.ai). 

Architecture: fastai unet_learner built off of resnet34, meaning an autoencoder with decoder portion from resnet34. Should switch to resnet50. `blur=True` is necessary to avoid the checkerboard effects (very obvious). Loss fn is straight outta style transfer (gatys et al), content loss plus gram matrix style loss. 


### training notes
notebook: KrisFace-GAN-v2.ipynb
In effect, I'm training an autoencoder whose first step is not an NN, but facial landmark extraction.

Dlib and a pretrained facial landmarks model allow the creation of identity facial landmarks from a video. To have an effective model that will create inferences off of landmarks from another person means transforms required on the identity landmarks during training.

__TODO:__ To that end, it would be wise to use face detection to return a bounding box of the face, so predictions are on a uniformly zoomed in distribution.

Ideal data would be an alignment between facial landmarks from person A's image and person B. In this case an algo wouldn't really be necessary since this is an actual mapping. Here we're training since facial pose is encoded (hopefully) w/in the landmarks. However landmarks also encode facial geometries which the end model needs to be independent of, e.g. width between the eyes, size of jaw, etc. __TODO:__ create symmetric warpings of the facial landmarks to make the final model more receptive of non-identity landmarks.

__TODO:__ train with different layers for the loss fn, train with resnet50. 

### Loss FN
Currently the style loss comes from the gram matrix difference at the second, third, and fourth of the ReLUs before MaxPool (Gatys recommends switching to avgpool if I recall). These losses are weighted toward three and four. Additionally the style loss has a weight compared to the content loss (content and style used in terms of the Style Transfer paper, all of the losses come from the same input img). Content loss comes from MSE applied to the output layer, y_hat, versus ground truth and the same layers as style loss with the same per-layer weighting.

Do I even need a validation set? Wouldn't validation set be based on landmarks from person B - no, since those landmarks don't have same facial pose