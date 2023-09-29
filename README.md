# Physics-informed CVAE for galaxy analysis 
A convolutional variational auto-encoder trained to extract physical galaxy features from multi-band imaging. The difference between this network and a vanilla CVAE is simple: an additional MSE loss term is added that guides the latent features toward the orientation, spectroscopic redshift, stellar mass, and star-formation rate of the galaxy. Weights are added to the various terms to prioritize learning these properties.

All the data can be downloaded from https://drive.google.com/drive/folders/1OvZzLBkdroyvzb0ano4GkAGL4FmOON45?usp=sharing in order to reproduce the results in the associated paper. 

The notebook `getRotations.ipynb` is used to calculate the orientation of each galaxy in the sample.

Then, the physics-informed VAE can be trained with the script `trainVAE_wRotAngle.py` and the vanilla VAE can be trained with the script `trainVAE_uninformed.py`.

Finally, the plots in the paper were generated with the notebook `findAnomalies.ipynb`. 

For questions, comments, and concerns, feel free to reach out at gaglian2@mit.edu.
