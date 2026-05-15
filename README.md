# Towards Unified Surgical Scene Understanding: Bridging Reasoning and Grounding via MLLMs

## MICCAI 2026 Early Accepted

This work has been provisionally accepted to MICCAI 2026 as an Early Accept paper, placing in the top 9% of submissions.

![Representative Tasks in Surgical Scene Understanding](images/background1.png)

![SurgMLLM Framework](images/Framework.png)

## Abstract

Surgical scene understanding is a cornerstone of computer-assisted intervention. While recent advances, particularly in surgical image segmentation, have driven progress, real-world clinical applications require a more holistic understanding that jointly captures procedural context, semantic reasoning, and precise visual grounding. However, existing approaches typically address these components in isolation, leading to fragmented representations and limited semantic consistency. To address this limitation, we propose SurgMLLM, a unified surgical scene understanding framework that bridges high-level reasoning and low-level visual grounding within a single model. Given surgical videos, SurgMLLM fine-tunes a multi-modal large language model (MLLM) to support structured interpretability reasoning, which is used to jointly model phases, instrument-verb-target ($IVT$) triplets, and triplet-entity segmentation tokens. These tokens are then temporally aggregated and serve as prompts for a segmentation network, enabling accurate pixel-wise grounding of triplet instruments and targets. The entire framework is trained end-to-end with a unified objective that couples language-based reasoning supervision with visual grounding losses, promoting coherent cross-task learning and clinically consistent scene representations. To facilitate unified evaluation, we introduce CholecT45-Scene, extending CholecT45 dataset with 64,299 frames of pixel-level mask annotations for instruments and targets, aligned with existing triplet labels. Extensive experiments show that SurgMLLM significantly advances surgical scene understanding, improving the primary triplet recognition metric AP$_{IVT}$ from 40.7\% to 46.0\% and consistently outperforming prior methods in phase recognition and segmentation. These results highlight the effectiveness of unified reasoning-and-grounding for reliable, context-aware surgical assistance. The code and dataset will be released.

## Code

We will release the code soon.

## Citation (Coming soon)

If you find this work useful, please consider citing our paper. The citation information will be updated soon.


