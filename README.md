## Score-based Density Estimation from Pairwise Comparisons<br><sub>Official PyTorch implementation of the ICLR 2026 paper</sub>

![Teaser image](./image.jpg)

**Score-based Density Estimation from Pairwise Comparisons**<br>
Petrus Mikkola, Luigi Acerbi, Arto Klami
<br>https://arxiv.org/abs/2510.09146<br>

Abstract: *We study density estimation from pairwise comparisons, motivated by expert knowledge elicitation and learning from human feedback. We relate the unobserved target density to a tempered winner density (marginal density of preferred choices), learning the winner's score via score-matching. This allows estimating the target by `de-tempering' the estimated winner density's score. We prove that the score vectors of the belief and the winner density are collinear, linked by a position-dependent tempering field. We give analytical formulas for this field and propose an estimator for it under the Bradley--Terry model. Using a diffusion model trained on tempered samples generated via score-scaled annealed Langevin dynamics, we can learn complex multivariate belief densities of simulated experts, from only hundreds to thousands of pairwise comparisons.*

## Requirements

See ...

## License & Attribution

This project is primarily original work by Petrus Mikkola (petrus-mikkola). However, because it incorporates and adapts code from a project under the **CC BY-NC-SA 4.0** license, this entire repository is distributed under the same terms to remain compliant with the "ShareAlike" requirement.

This repository is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**. 
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/)

---

### Credits and Derivative Works

This project builds upon and modifies code from the following sources:

1.  **NVIDIA CORPORATION & AFFILIATES**
    * **Source:** [Repository](https://github.com/NVlabs/edm)
    * **License:** [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/)
    * **Contribution:** Modified code from xxx. Under the **ShareAlike (SA)** clause, this repository adopts this license.

3.  **Google Research Authors**
    * **Source:** [Repository](https://github.com/yang-song/score_sde_pytorch)
    * **License:** [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)
    * **Contribution:** Modified code from likelihood.py. 


## Citation

```
@inproceedings{xxx
}
```
