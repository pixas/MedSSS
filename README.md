# MedSSS

<p align="center">
📃 <a href="http://arxiv.org/abs/2501.12051" target="_blank">Paper</a> ｜🤗 <a href="https://huggingface.co/pixas/MedSSS_Policy" target="_blank">MedSSS-8B-Policy</a> ｜🤗 <a href="https://huggingface.co/pixas/MedSSS_PRM" target="_blank">MedSSS-8B-PRM</a> | 📚 <a href="https://huggingface.co/datasets/pixas/MedSSS-data" target="_blank">SFT/PRM Data</a>
</p>

## 💫 News
- 🔥 [2025/11/08] MedS$^3$ has been accepted as a poster at AAAI 2026 Main!

## ⚡Introduction
This repository contains the self-evolving pipeline of MedS$^3$, a slow-thinking small medical language models built with a self-evolution pipeline and an innovative soft dual-sided process supervision.

<div align=center>
<img src="assets/framework.png"  width = "90%" alt="MedSSS" align=center/>
</div>


**MedS$^3$** is a medical LLM designed for advanced medical reasoning, with reliable intermediate reasoning steps. It can leverage the PRM model to select the most correct response from several outputs. It supports both traditional medical question answering problems, as well as realistic clinical problems.
It is built with the following three steps

- Using Monte-Carlo Tree Search to self-collect correct and incorrect reasoning trajectories.
- Use SFT to train a policy model in the correct trajectory set and use soft-label two-class classification to train a PRM model in both correct/incorrect internal reasoning steps.
- Use PRM best-of-N decoding method to generate several candidate responses and use PRM to select the most appropriate one, with the highest PRM score.

We open-sourced our models, data, and code here.





## 👨‍⚕️ Model
- **Model Access**

|                      | Backbone     | Supported Languages | Link                                                                  |
| -------------------- | ------------ | ----- | --------------------------------------------------------------------- |
| **MedSSS-8B-Policy**  | LLaMA-3.1-8B  | English    | [HF Link](https://huggingface.co/pixas/MedSSS_Policy) |
| **MedSSS-8B-PRM**  | LLaMA-3.1-8B  | English    | [HF Link](https://huggingface.co/pixas/MedSSS_PRM) |

Please follow the Huggingface page to deploy the two models





## 📚 Data
- **Data Access**

You can access the detailed step-by-step solution in [HF link](https://huggingface.co/datasets/pixas/MedSSS-data).
