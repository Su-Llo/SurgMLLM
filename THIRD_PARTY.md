# Third-party software and provenance

SurgMLLM incorporates the upstream components and engineering patterns listed
below. Project-specific code is Apache-2.0 licensed, while each upstream
component remains subject to its respective license and notice.

| Component | Upstream | Use in this repository | License / notice |
|---|---|---|---|
| Sa2VA | ByteDance Seed, `ByteDance/Sa2VA` | MMEngine/XTuner project layout, direct four-step train→convert→infer→evaluate shape, language-prompted SAM2 integration lineage | Apache-2.0; retain upstream authorship |
| Segment Anything 2 | Meta Platforms, `facebookresearch/sam2` | Hiera-Large image encoder, prompt encoder, mask decoder, and supporting model graph under `third_parts/sam2`; the HF export uses the corresponding minimal runtime | Apache-2.0; source files retain Meta copyright headers |
| InternVL2.5 | OpenGVLab, `OpenGVLab/InternVL2_5-4B` | Upstream multimodal backbone and remote-code files needed by a converted model | MIT; see `LICENSE_InternVL`; source files retain OpenGVLab headers; weights are not redistributed by Git |
| FastChat | LMSYS Org, `lm-sys/FastChat` | Conversation prompt template adapted in `projects/surgmllm/hf/conversation.py` | Apache-2.0; derivative notice retained in source |
| MMEngine | OpenMMLab | Runner, hooks, configuration, checkpoints | Apache-2.0 |
| MMDetection | OpenMMLab | Technical precursor for the original loss/sample-point path; the current runtime uses native PyTorch BCE/Dice and does not vendor MMDetection source | Apache-2.0 |
| XTuner | OpenMMLab | InternVL wrapper, distributed training entry, ZeRO-2 integration | Apache-2.0 |
| Hugging Face Transformers / PEFT | Hugging Face | Model loading, generation, LoRA | Apache-2.0 |

No dataset, checkpoint, converted model, prediction, visualization, or upstream
model weight is tracked in Git. Users must obtain CholecT data and the two
upstream initialization checkpoints under their respective terms.
