# Related work

Short, sourced notes for a paper-style related-work section. Links point at
the archival version we checked. This is not a complete survey.

## Datasets

- **CIC-IDS2017.** Sharafaldin, Lashkari, Ghorbani. “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.” ICISSP 2018, pp. 108–116. [DOI 10.5220/0006639801080116](https://doi.org/10.5220/0006639801080116). Official page: [unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html).
- **CSE-CIC-IDS2018.** Same group; AWS distribution at [registry.opendata.aws/cse-cic-ids2018](https://registry.opendata.aws/cse-cic-ids2018/). Page: [unb.ca/cic/datasets/ids-2018.html](https://www.unb.ca/cic/datasets/ids-2018.html).
- **UNSW-NB15.** Moustafa and Slay. “UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set).” MilCIS 2015. [DOI 10.1109/MilCIS.2015.7348942](https://doi.org/10.1109/MilCIS.2015.7348942). [Project page](https://research.unsw.edu.au/projects/unsw-nb15-dataset).
- **CIC-IoT-2023.** Neto et al. “CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment.” *Sensors* 23(13):5941, 2023. [DOI 10.3390/s23135941](https://doi.org/10.3390/s23135941).

**Dataset hygiene.** Engelen, Rimmer, Joosen. “Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study.” IEEE Security & Privacy Workshops 2021, pp. 7–12. [DOI 10.1109/SPW53761.2021.00009](https://doi.org/10.1109/SPW53761.2021.00009). They show CICFlowMeter / labelling errors that change downstream ML rankings and release a reconstructed feature set: [intrusion-detection.distrinet-research.be/WTMC2021](https://intrusion-detection.distrinet-research.be/WTMC2021/). Any serious CICIDS paper should cite this and say which dump was used.

## Gradient attacks and adversarial training

- Szegedy et al. “Intriguing properties of neural networks.” ICLR 2014. [arXiv:1312.6199](https://arxiv.org/abs/1312.6199).
- Goodfellow, Shlens, Szegedy. “Explaining and Harnessing Adversarial Examples.” ICLR 2015. [arXiv:1412.6572](https://arxiv.org/abs/1412.6572). **FGSM.**
- Madry et al. “Towards Deep Learning Models Resistant to Adversarial Attacks.” ICLR 2018. [arXiv:1706.06083](https://arxiv.org/abs/1706.06083). **PGD** and PGD adversarial training.
- Zhang et al. “Theoretically Principled Trade-off between Robustness and Accuracy.” ICML 2019. [arXiv:1901.08573](https://arxiv.org/abs/1901.08573). **TRADES** (optional upgrade over vanilla adv-training).
- Papernot, McDaniel, Goodfellow. “Transferability in Machine Learning: from Phenomena to Black-Box Attacks using Adversarial Samples.” 2016. [arXiv:1605.07277](https://arxiv.org/abs/1605.07277).

This repo’s FGSM/PGD are the standard first-order methods with an added
**feature mask + [0,1] box**, i.e. they are *constrained* feature-space
attacks, not ImageNet copies.

## GAN-based adversarial examples

- Goodfellow et al. “Generative Adversarial Nets.” NeurIPS 2014.
- Xiao, Li, Zhu, He, Liu, Song. “Generating Adversarial Examples with Adversarial Networks.” IJCAI 2018, pp. 3905–3911. [DOI 10.24963/ijcai.2018/543](https://doi.org/10.24963/ijcai.2018/543) (**AdvGAN**): generator produces a perturbation, discriminator keeps samples on-manifold, a frozen classifier supplies the fooling loss. Semi-white-box after the generator is trained. Our conditional GAN is this recipe on flow vectors.

## GAN evasion against IDS

- Lin, Shi, Xue. “IDSGAN: Generative Adversarial Networks for Attack Generation against Intrusion Detection.” PAKDD 2022, LNCS 13282, pp. 79–91. Preprint [arXiv:1809.02077](https://arxiv.org/abs/1809.02077). WGAN generator + a discriminator that *imitates a black-box IDS*; **restricted modification** of functional features. Evaluated on NSL-KDD.
- Hou, Lv, Li, et al. “DEMGAN: A Machine Learning-Based Intrusion Detection System Evasion Scheme.” *Computers, Materials & Continua* 84(1), 2025. [DOI 10.32604/cmc.2025.064833](https://doi.org/10.32604/cmc.2025.064833). Distortion-enhanced multi-generator WGAN on **CICIDS2017 and CICIDS2018**; they also retrain the IDS on generated rows. Treat published evasion percentages as *their* numbers, not ours.
- Shu, Wan, Li, et al. “Generating Practical Adversarial Network Traffic Flows Using NIDSGAN.” 2022. [arXiv:2203.06694](https://arxiv.org/abs/2203.06694). Domain constraints + limited oracle access; white-box / black-box / restricted-black-box numbers on DNN NIDS.

Usama, Asim, Latif, Qadir, Ala-Al-Fuqaha and follow-ups study GANs both to
*attack* and to *augment* NIDS; cite the specific paper you compare against
rather than a vague “GAN IDS” bucket.

## Constrained / domain-valid perturbations

Network features are not pixels. Independent work makes that precise:

- Sheatsley, Papernot, Weisman, Verma, McDaniel. “Adversarial Examples in Constrained Domains.” [arXiv:2011.01183](https://arxiv.org/abs/2011.01183). Journal version: “Adversarial examples for network intrusion detection systems,” *Journal of Computer Security* 30(5), 2022. [DOI 10.3233/JCS-210094](https://doi.org/10.3233/JCS-210094). Adaptive JSMA + Histogram Sketch Generation on NSL-KDD / UNSW-NB15 under domain rules. Finding: a tiny allowed feature set is still enough to evade.
- Pierazzi, Pendlebury, Cortellazzi, Cavallaro. “Intriguing Properties of Adversarial ML Attacks in the Problem Space.” IEEE S&P 2020. [DOI 10.1109/SP40000.2020.00073](https://doi.org/10.1109/SP40000.2020.00073). Feature-space success ≠ a valid problem-space object. Our README states this limit explicitly.
- Hashemi, Cusack, Keller. “Towards Evaluation of NIDSs in Adversarial Setting.” Big-DAMA @ SIGCOMM 2019. [DOI 10.1145/3345619.3349938](https://doi.org/10.1145/3345619.3349938).

## Threat models for ML-NIDS

- Apruzzese, Andreolini, Ferretti, Marchetti, Colajanni. “Modeling Realistic Adversarial Attacks against Network Intrusion Detection Systems.” *Digital Threats: Research and Practice* 3(3), Article 31, 2022. [DOI 10.1145/3469659](https://doi.org/10.1145/3469659). Five “power” axes; many papers assume oracles that a remote attacker does not have. We use that language in [THREAT_MODEL.md](THREAT_MODEL.md).
- Alhajjar, Maxwell, Bastian. “Adversarial machine learning in Network Intrusion Detection Systems.” *Expert Systems with Applications* 186, 2021. [DOI 10.1016/j.eswa.2021.115742](https://doi.org/10.1016/j.eswa.2021.115742). Broader AML-on-NIDS survey.

## How this repo sits in that map

| Line of work | What we reuse | What we do not claim |
| --- | --- | --- |
| AdvGAN / IDSGAN / DEMGAN | Conditional generator, frozen IDS, functional-feature mask, evasion rate | WGAN-GP, multi-generator DEMGAN, NSL-KDD numbers |
| FGSM / PGD / Madry | Constrained first-order attacks + PGD adversarial training | TRADES / certified defenses |
| Sheatsley / Pierazzi | Mask + box + “feature space ≠ packets” | Adaptive JSMA, problem-space packet mutation |
| Apruzzese | White-box / grey-box / transfer cells | A claim that white-box PGD is operationally realistic |

When you write the paper, **compare against constrained PGD and a reimplemented
IDSGAN-style GAN on the same splits** before claiming a new generator is
better. DEMGAN’s headline evasion rates are not reproduced here.
