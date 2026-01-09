# $\beta$-Diffusions Learns High-level Visual Representations

We propose $\beta$-Diffusion, a novel generative framework that integrates the controllable noise regimes of diffusion models into a Variational Autoencoder (VAE) architecture. Unlike traditional regularized Autoencoders (rAE), $\beta$-Diffusion employs a Vision Transformer (ViT) encoder and a VAE Modulator powered by Diffusion Transformer (DiT) blocks. The modulator dynamically adjusts the latent distribution parameters, $\mu$ and $\sigma$, as a function of a continuous noise scale $\beta$. By training a lightweight decoder to reconstruct images across a stochastic range of $\beta \sim U(1, 100)$, the model learns a noise-aware latent manifold. This approach enables precise control over the information bottleneck, allowing for flexible trade-offs between reconstruction fidelity and generative prior alignment.


To refine your mathematical notation, we need to ensure the relationship between the modulation blocks and the latent parameters is precise. For the **Related Works**, we will position your idea at the intersection of Variational Inference and the recent trend of using Transformers as backbone modulators.



## Method
### Feature Extraction and Modulation

Let $E$ be the ViT encoder and $M$ be the VAE modulator (DiT). Given an input $x$, the hidden representation is $h = E(x)$. The modulator $M$ outputs modulation parameters conditioned on the noise level $\beta$:

$$
\mu = f_{\mu}(h) * s_{\mu}(h;\beta), \\
\sigma = f_{\sigma}(h) * s_{\sigma}(h;\beta),
$$
where $f_{\mu}$ and $f_{\sigma}$ are linear projections that map $h$ to the latent dimension, and $s_{\mu}$ and $s_{\sigma}$ are the outputs of the **Diffusion Transformer (DiT)** blocks, representing the $\beta$-dependent scale/shift factors.


### The Objective Function

The training objective is a -weighted Evidence Lower Bound (ELBO). For a given pair , the loss is:

$$
L_{\text{ELBO}} = \|x - D_\psi(z, \beta)\|_2^2 + \beta \cdot \mathrm{KL}(q(z|x;\beta) \| p(z)),
$$
Here, $D_\psi$ is the MLP decoder, which also receives $\beta$ as an auxiliary input to adapt its reconstruction strategy to the enforced bottleneck intensity.

## Experiments
```bash
python -m debugpy --listen 0.0.0.0:5679 -m torch.distributed.launch src/train_beta_diffusion.py \
  --config configs/stage1/training/BetaDiffusion_DINOv2-B.yaml \
  --data-path $IMNET 

ENTITY=erow PROJECT='beta-diffusion' EXPERIMENT_NAME=mae-trail_v0.1 sbatch  ~/storchrun.sh src/train_beta_diffusion.py  --data-path $IMNET  --config configs/stage1/training/BetaDiffusion_MAE-B.yaml --wandb
```


## Comparison

| Feature         | β-VAE                   | Latent Diffusion (LDM)      | β-Diffusion (Ours)          |   |
|-----------------|-------------------------|-----------------------------|-----------------------------|---|
| Bottleneck Type | Static / Fixed $\beta$  | Fixed VAE + Diffusion Prior | Dynamic / Modulated $\beta$ |   |
| Backbone        | Usually CNN             | UNet or DiT                 | ViT Encoder + DiT Modulator |   |
| Noise Handling  | Constant regularization | Iterative denoising steps   | Single-step adaptive noise  |   |
| Inference Mode  | Single forward pass     | Many denoising iterations   | Variable-intensity pass     |   |
| Encoder Goal    | Identity mapping + Reg. | Dimensionality reduction    | Noise-conditioned encoding  |   |
| Role of $\beta$ | Hyperparameter          | Not directly applicable     | Input-conditioned variable  |   |