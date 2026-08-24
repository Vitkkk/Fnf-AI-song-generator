# Latent-space diagnostic — v0.1 tiny CPU

This diagnostic was run on the best checkpoint from the first 50-epoch experiment (epoch 35, validation loss 0.1636857788).

## Question

The random generated samples sounded mostly like very low bass / 808-like energy plus whispery noise. The diagnostic tests whether the decoder itself failed to learn useful structure, or whether samples from the standard normal prior `N(0, 1)` are landing outside the latent regions the encoder actually uses.

## Held-out reconstruction test

A deterministic validation crop from `hi_jon` (which was excluded from training) was encoded and decoded using the posterior mean `mu`.

| Metric | Reconstruction from `mu` | Random prior sample |
| --- | ---: | ---: |
| mel MAE vs held-out target | **0.160789** | 0.332736 |
| mel correlation vs held-out target | **0.694701** | 0.182335 |

The reconstruction error is less than half the random-prior error, and the reconstructed spectrogram retains substantial correlation with a song the model never trained on.

## Posterior/prior mismatch

Across all precomputed train and validation crops:

- aggregate `mu` mean: -0.0156
- aggregate `mu` standard deviation: 1.1659
- mean posterior standard deviation: **0.1552**
- median posterior standard deviation: 0.1511
- mean posterior `mu` L2 norm: 6.4825
- expected L2 norm of a 32-D `N(0,1)` sample: 5.6569
- mean summed KL: 66.7617

The most important value is the posterior standard deviation. The encoder places each example in a relatively tight neighborhood (`sigma ~= 0.155`) while generation samples with unit standard deviation from the prior. This makes the decoder see thin learned regions during training but arbitrary space during generation.

## Nearest-manifold distance

Using encoded training-crop posterior means as a simple approximation of the learned latent manifold:

- train crop -> nearest other train crop: mean distance **3.58**
- held-out validation crop -> nearest train crop: mean distance **3.47**
- random `N(0,1)` point -> nearest train crop: mean distance **6.81**
- the exact seed used for the original noisy generation -> nearest train crop: **7.25**

So the original generated seed landed roughly twice as far from the learned latent regions as a real held-out example does.

## Extra interpolation test

A new latent point was created by interpolating between the song-centroid latents of `maniac` and `lost_latinas_part2`. Their centroid distance was 1.5903. This is not a direct reconstruction of the held-out `hi_jon` crop; it is a new point built inside the geometry learned by the encoder.

## Conclusion

The first noisy generation should not be interpreted as evidence that the decoder learned nothing. The evidence supports a posterior/prior mismatch:

1. the encoder-decoder reconstructs held-out musical spectrogram structure substantially better than a random prior sample;
2. validation latent points occupy the same neighborhoods as training points;
3. standard-normal random samples land much farther from those neighborhoods;
4. posterior variance is far narrower than the generation prior.

For the next experiment, the highest-value changes are to improve prior matching (KL schedule / stronger KL, richer prior, or aggregate-posterior sampling) and then move toward a temporal latent model / learned codec for actual musical generation quality.
