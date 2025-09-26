# Awesome RL for Large Reasoning Models [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A comprehensive collection of research papers on **reinforcement learning (RL) techniques for enhancing reasoning capabilities** in LLMs, VLMs, and MLLMs.

This repository focuses on the intersection of reinforcement learning and reasoning capabilities, covering mathematical reasoning, code generation, multimodal reasoning, and specialized domain applications. 
## Table of Contents

- [Survey Papers](#survey-papers)
- [Foundational Papers](#foundational-papers)
- [GRPO (Group Relative Policy Optimization)](#grpo-group-relative-policy-optimization)
- [R1 Algorithms and Variants](#r1-algorithms-and-variants)
- [Traditional RL Approaches](#traditional-rl-approaches)
- [RL for LLMs](#rl-for-llms)
- [RL for MLLMs](#rl-for-mllms)
- [Benchmarks and Evaluation](#benchmarks-and-evaluation)
- [Open-Source Projects](#open-source-projects)

---

## Survey Papers

### Comprehensive Surveys
- **A Survey of Reinforcement Learning for Large Reasoning Models** [[paper](https://arxiv.org/abs/2509.08827)]

- **A Technical Survey of RL Techniques for Large Language Models**  [[paper](https://arxiv.org/abs/2507.04136)]

- **Reinforcement Fine-Tuning Powers Reasoning Capability of Multimodal Large Language Models** [[paper](https://arxiv.org/abs/250518536)]

---

## Foundational Papers

Core papers that established the field of reinforcement learning for language model alignment and reasoning:

### RLHF Foundations
- **Deep Reinforcement Learning from Human Preferences** - [arXiv:1706.03741](https://arxiv.org/abs/1706.03741) *(NeurIPS 2017)*  
  *Christiano et al.* - Seminal work establishing human preference learning for RL, laying groundwork for all RLHF research

- **Training language models to follow instructions with human feedback** - [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) *(NeurIPS 2022)*  
  *Ouyang et al. (OpenAI)* - The InstructGPT paper that established RLHF as standard alignment technique, showing 1.3B model preferred over 175B GPT-3

- **Constitutional AI: Harmlessness from AI Feedback** - [arXiv:2212.08073](https://arxiv.org/abs/2212.08073) *(arXiv 2022)*  
  *Bai et al. (Anthropic)* - Introduced RLAIF as alternative to RLHF using AI-generated preferences and constitutional principles

### Recent Breakthrough
- **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** - [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) *(Nature 2025)*  
  *DeepSeek-AI* - **Major breakthrough:** First reasoning model trained via pure RL without supervised fine-tuning, achieving o1-1217 level performance

---

## GRPO (Group Relative Policy Optimization)

### Core GRPO Papers
- **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** - [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) *(arXiv 2024)*  
  *DeepSeek-AI* - **Original GRPO paper:** Introduced GRPO as memory-efficient PPO alternative, eliminating critic model, achieving 51.7% on MATH

### GRPO Variants and Extensions
- **Hybrid Group Relative Policy Optimization: A Multi-Sample Approach** - [arXiv:2502.01652](https://arxiv.org/abs/2502.01652) *(arXiv 2025)*  
  *Multi-sample approach balancing empirical action sampling with value function learning*

- **Stepwise Guided Policy Optimization: Coloring your Incorrect Reasoning in GRPO** - [arXiv:2505.11595](https://arxiv.org/abs/2505.11595) *(arXiv 2025)*  
  *Addresses GRPO's all-negative-sample limitation using step-wise judge models*

- **MAPO: Mixed Advantage Policy Optimization** - [arXiv:2509.18849](https://arxiv.org/abs/2509.18849) *(arXiv 2024)*  
  *Solves advantage reversion and advantage mirror problems in GRPO with dynamic reweighting*

- **Geometric-Mean Policy Optimization** - [arXiv:2507.20673](https://arxiv.org/abs/2507.20673) *(arXiv 2025)*  
  *Stabilized GRPO variant using geometric mean instead of arithmetic mean, 4.1% improvement on math benchmarks*

- **Revisiting Group Relative Policy Optimization: Insights into On-Policy and Off-Policy Training** - [arXiv:2505.22257](https://arxiv.org/abs/2505.22257) *(arXiv 2025)*  
  *Extends GRPO to off-policy settings with theoretical analysis*

### GRPO Applications

#### Mathematical Reasoning
- **GRPO-LEAD: A Difficulty-Aware Reinforcement Learning Approach** - [arXiv:2504.09696](https://arxiv.org/abs/2504.09696) *(arXiv 2025)*  
  *Difficulty-aware GRPO with length-dependent accuracy rewards for concise mathematical reasoning*

#### Code Generation
- **Posterior-GRPO: Rewarding Reasoning Processes in Code Generation** - [arXiv:2508.05170](https://arxiv.org/abs/2508.05170) *(arXiv 2025)*  
  *Addresses reward hacking in code generation with process-based rewards, 4.5% improvement over baselines*

- **From Reasoning to Code: GRPO Optimization for Underrepresented Languages** - [arXiv:2506.11027](https://arxiv.org/abs/2506.11027) *(arXiv 2025)*  
  *GRPO for underrepresented programming languages like Prolog*

#### Multimodal Applications
- **MM-R1: Unleashing the Power of Unified Multimodal LLMs for Personalized Image Generation** - [arXiv:2508.11433](https://arxiv.org/abs/2508.11433) *(arXiv 2024)*  
  *GRPO for cross-modal chain-of-thought reasoning in personalized image generation*

- **AlphaMaze: Enhancing LLMs' Spatial Intelligence via GRPO** - [arXiv:2502.14669](https://arxiv.org/abs/2502.14669) *(arXiv 2025)*  
  *Visual spatial reasoning for maze navigation achieving 93% accuracy*

---

## R1 Algorithms and Variants

### Core R1 Framework
- **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** - [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) *(Nature 2025)*  
  *Foundational R1 approach using pure RL without SFT, comparable to OpenAI-o1-1217*

### Mathematical Reasoning R1 Variants
- **Training Large Language Models to Reason via EM Policy Gradient** - [arXiv:2504.18587](https://arxiv.org/abs/2504.18587) *(arXiv 2025)*  
  *Expectation-Maximization framing of reasoning as alternative to PPO/GRPO*

### Advanced Reasoning Frameworks
- **Parallel-R1: Towards Parallel Thinking via Reinforcement Learning** - [arXiv:2509.07980](https://arxiv.org/abs/2509.07980) *(arXiv 2025)*  
  *First RL framework enabling parallel thinking behaviors, 8.4% accuracy improvement over sequential models*

- **Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning** - [arXiv:2502.14768](https://arxiv.org/abs/2502.14768) *(arXiv 2025)*  
  *Uses synthetic logic puzzles for training, 125% improvement on AIME with only 5K training examples*

### Multimodal R1 Systems
- **R1-VL: Learning to Reason with Multimodal LLMs via Step-wise GRPO** - [arXiv:2503.12937](https://arxiv.org/abs/2503.12937) *(arXiv 2025)*  
  *First R1-style training for vision-language models with step-wise dense rewards*

- **Vision-R1: Incentivizing Reasoning Capability in Multimodal LLMs** - [arXiv:2503.06749](https://arxiv.org/abs/2503.06749) *(arXiv 2025)*  
  *Step-wise GRPO for multimodal mathematical reasoning, 73.5% on MathVista*

### Agent-Based R1 Systems
- **WebAgent-R1: Training Web Agents via End-to-End Multi-Turn RL** - [arXiv:2505.16421](https://arxiv.org/abs/2505.16421) *(arXiv 2025)*  
  *Web navigation: Qwen-2.5-3B: 6.1% → 33.9%, Llama-3.1-8B: 8.5% → 44.8% success rates*

- **Agent-R1: End-to-End Agent Training Framework** - [GitHub](https://github.com/0russwest0/Agent-R1) *(Open Source)*  
  *Open-source framework for training LLM agents using end-to-end RL*

---

## Traditional RL Approaches

### Proximal Policy Optimization (PPO)
- **DialCoT Meets PPO: Decomposing and Exploring Reasoning Paths** - [ACL 2023](https://aclanthology.org/2023.emnlp-main.501/) *(EMNLP 2023)*  


- **Teaching Large Language Models to Reason with Reinforcement Learning** - [arXiv:2403.04642](https://arxiv.org/abs/2403.04642) *(arXiv 2024)*  


### Direct Preference Optimization (DPO)
- **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** - [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) *(arXiv 2023)*  


- **Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning** - [arXiv:2406.18629](https://arxiv.org/abs/2406.18629) *(arXiv 2024)*  


### Actor-Critic Methods
- **Enhancing Decision-Making of LLMs via Actor-Critic** - [OpenReview](https://openreview.net/forum?id=0tXmtd0vZG) *(ICLR 2024)*  


- **PACE: Improving Prompt with Actor-Critic Editing for LLMs** - [arXiv:2308.10088](https://arxiv.org/abs/2308.10088) *(arXiv 2023)*  


### Tree Search + RL Combinations
- **TreeRL: LLM Reinforcement Learning with On-Policy Tree Search** - [ACL 2025](https://aclanthology.org/2025.acl-long.604/) *(ACL 2025)*  


- **Alphazero-like Tree-Search can Guide LLM Decoding and Training** - [arXiv:2309.17179](https://arxiv.org/abs/2309.17179) *(ICML 2024)*  

- **Policy Guided Tree Search for Enhanced LLM Reasoning** - [arXiv:2502.06813](https://arxiv.org/abs/2502.06813) *(arXiv 2025)*  


### REINFORCE and Policy Gradients
- **Reinforcement Learning for Reasoning in LLMs with One Training Example** - [arXiv:2504.20571](https://arxiv.org/abs/2504.20571) *(arXiv 2024)*  

### Process Reward Models
- **The Lessons of Developing Process Reward Models in Mathematical Reasoning** - [arXiv:2501.07301](https://arxiv.org/abs/2501.07301) *(arXiv 2025)*  

- **Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning** - [OpenReview](https://openreview.net/forum?id=A6Y7AqlzLW) *(ICLR 2025)*  

---
## RL for LLMs

* DAPO: an Open-Source LLM Reinforcement Learning System at Scale | [[Paper]](https://arxiv.org/pdf/2503.14476) | [[GitHub]](https://github.com/BytedTsinghua-SIA/DAPO)


---

## RL for MLLMs

### Vision-Language Models (VLMs)
* [2508] [InternVL3.5] [InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency](http://arxiv.org/abs/2508.18265)  [[Model](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B)]  [[Code](https://github.com/OpenGVLab/InternVL)]  

* [2508] [Thyme] [Thyme: Think Beyond Images](https://arxiv.org/abs/2508.11630)  [[Project](https://thyme-vl.github.io/)]  [[Models](https://huggingface.co/collections/Kwai-Keye/thyme-689ebea74a628c3a9b7bd789)]  [[Datasets](https://huggingface.co/collections/Kwai-Keye/thyme-689ebea74a628c3a9b7bd789)]  [[Code](https://github.com/yfzhang114/Thyme)]  

* [2508] [MM-R1 (generation)] [MM-R1: Unleashing the Power of Unified Multimodal Large Language Models for Personalized Image Generation](https://arxiv.org/abs/2508.11433)

* [2508] [We-Math 2.0] [We-Math 2.0: A Versatile MathBook System for Incentivizing Visual Mathematical Reasoning](https://arxiv.org/abs/2508.10433) [[Project](https://we-math2.github.io/)]  [[Datasets](https://huggingface.co/datasets/We-Math/We-Math2.0-Standard)] [[Code](https://github.com/We-Math/We-Math2.0)] 

* [2508] [Skywork UniPic 2.0] [Skywork UniPic 2.0: Building Kontext Model with Online RL for Unified Multimodal Model](https://github.com/SkyworkAI/UniPic/blob/main/UniPic-2/assets/pdf/UNIPIC2.pdf)  [[Project](https://unipic-v2.github.io/)]  [[Models](https://huggingface.co/collections/Skywork/skywork-unipic2-6899b9e1b038b24674d996fd)]  [[Code](https://github.com/SkyworkAI/UniPic/tree/main/UniPic-2)] 

* [2508] [DocThinker] [DocThinker: Explainable Multimodal Large Language Models with Rule-based Reinforcement Learning for Document Understanding](https://arxiv.org/abs/2508.08589) [[Code](https://github.com/wenwenyu/DocThinker)] 

* [2508] [AR-GRPO (generation)] [AR-GRPO: Training Autoregressive Image Generation Models via Reinforcement Learning](https://arxiv.org/abs/2508.06924) [[Models](https://huggingface.co/collections/CSshihao/ar-grpo-689c970f4c848f01a162352a)]  [[Code](https://github.com/Kwai-Klear/AR-GRPO)] 

* [2508] [M2IO-R1] [M2IO-R1: An Efficient RL-Enhanced Reasoning Framework for Multimodal Retrieval Augmented Multimodal Generation](https://arxiv.org/abs/2508.06328) 

* [2508] [SIFThinker] [SIFThinker: Spatially-Aware Image Focus for Visual Reasoning](https://arxiv.org/abs/2508.06259) 

* [2508] [Shuffle-R1] [Shuffle-R1: Efficient RL framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle](https://arxiv.org/abs/2508.05612) [[Code](https://github.com/XenoZLH/Shuffle-R1)] 

* [2508] [TempFlow-GRPO (generation)] [TempFlow-GRPO: When Timing Matters for GRPO in Flow Models](https://arxiv.org/abs/2508.04324) 

* [2508] [EARL (editing)] [The Promise of RL for Autoregressive Image Editing](https://arxiv.org/abs/2508.01119) [[Code](https://github.com/mair-lab/EARL)] 

* [2507] [VL-Cogito] [VL-Cogito: Progressive Curriculum Reinforcement Learning for Advanced Multimodal Reasoning](https://arxiv.org/abs/2507.22607) [[Model](https://huggingface.co/csyrf/VL-Cogito)] [[Dataset](https://huggingface.co/datasets/csyrf/VL-Cogito)] [[Code](https://github.com/alibaba-damo-academy/VL-Cogito)] 

* [2507] [X-Omni (generation)] [X-Omni: Reinforcement Learning Makes Discrete Autoregressive Image Generative Models Great Again](https://arxiv.org/abs/2507.22058) [[Project](https://x-omni-team.github.io/)] [[Models](https://huggingface.co/collections/X-Omni/x-omni-models-6888aadcc54baad7997d7982)] [[Dataset](https://huggingface.co/datasets/X-Omni/LongText-Bench)] [[Code](https://github.com/X-Omni-Team/X-Omni)] 

* [2507] [MixGRPO (generation)] [MixGRPO: Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE](https://arxiv.org/abs/2507.21802) [[Project](https://tulvgengenr.github.io/MixGRPO-Project-Page/)] [[Model](https://huggingface.co/tulvgengenr/MixGRPO)] [[Code](https://github.com/Tencent-Hunyuan/MixGRPO)]

* [2507] [RRVF] [Learning Only with Images: Visual Reinforcement Learning with Reasoning, Rendering, and Visual Feedback](https://arxiv.org/abs/2507.20766)  [[Model](https://huggingface.co/chenzju/rrvf_chartmimic)] [[Dataset](https://huggingface.co/datasets/syficy/rrvf_coldstart_chartqa)] [[Code](https://github.com/L-O-I/RRVF)]

* [2507] [SOPHIA] [Semi-off-Policy Reinforcement Learning for Vision-Language Slow-thinking Reasoning](https://arxiv.org/abs/2507.16814) 

* [2507] [Spatial-VLM-Investigator] [Enhancing Spatial Reasoning in Vision-Language Models via Chain-of-Thought Prompting and Reinforcement Learning](https://arxiv.org/abs/2507.13362)  [[Code](https://github.com/Yvonne511/spatial-vlm-investigator)]

* [2507] [VisionThink] [VisionThink: Smart and Efficient Vision Language Model via Reinforcement Learning](https://arxiv.org/abs/2507.13348) [[Models](https://huggingface.co/collections/Senqiao/visionthink-6878d839fae02a079c9c7bfe)]  [[Datasets](https://huggingface.co/collections/Senqiao/visionthink-6878d839fae02a079c9c7bfe)]  [[Code](https://github.com/dvlab-research/VisionThink)]

* [2507] [M2-Reasoning] [M2-Reasoning: Empowering MLLMs with Unified General and Spatial Reasoning](https://arxiv.org/abs/2507.08306)  [[Model](https://huggingface.co/inclusionAI/M2-Reasoning)]  [[Code](https://github.com/inclusionAI/M2-Reasoning)]

* [2507] [SFT-RL-SynergyDilemma] [The Synergy Dilemma of Long-CoT SFT and RL: Investigating Post-Training Techniques for Reasoning VLMs](https://arxiv.org/abs/2507.07562)   [[Models](https://huggingface.co/JierunChen)]  [[Datasets](https://huggingface.co/JierunChen)]  [[Code](https://github.com/JierunChen/SFT-RL-SynergyDilemma)]

* [2507] [PAPO] [PAPO: Perception-Aware Policy Optimization for Multimodal Reasoning](https://arxiv.org/abs/2507.06448)  [[Project](https://mikewangwzhl.github.io/PAPO/)]  [[Models](https://huggingface.co/collections/PAPOGalaxy/papo-qwen-686d92dd3d43b1ce698f851a)]  [[Datasets](https://huggingface.co/collections/PAPOGalaxy/data-686da53d67664506f652774f)]  [[Code](https://github.com/MikeWangWZHL/PAPO)]

* [2507] [Skywork-R1V3] [Skywork-R1V3 Technical Report](https://arxiv.org/abs/2507.06167)  [[Model](https://huggingface.co/Skywork/Skywork-R1V3-38B)]  [[Code](https://github.com/SkyworkAI/Skywork-R1V)]

* [2507] [Open-Vision-Reasoner] [Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning](https://arxiv.org/abs/2507.05255) [[Project](https://weiyana.github.io/Open-Vision-Reasoner/)] [[Models](https://huggingface.co/collections/Kangheng/ovr-686646849f9b43daccbe2fe0)]  [[Code](https://github.com/Open-Reasoner-Zero/Open-Vision-Reasoner)]

* [2507] [GLM-4.1V-Thinking] [GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning](https://arxiv.org/abs/2507.01006) [[Models](https://huggingface.co/collections/THUDM/glm-41v-thinking-6862bbfc44593a8601c2578d)] [[Demo](https://huggingface.co/spaces/THUDM/GLM-4.1V-9B-Thinking-API-Demo)] [[Code](https://github.com/THUDM/GLM-4.1V-Thinking)]

* [2506] [MiCo] [MiCo: Multi-image Contrast for Reinforcement Visual Reasoning](https://arxiv.org/abs/2506.22434) 

* [2506] [Visual-Structures] [Visual Structures Helps Visual Reasoning: Addressing the Binding Problem in VLMs](https://arxiv.org/abs/2506.22146) 

* [2506] [APO] [APO: Enhancing Reasoning Ability of MLLMs via Asymmetric Policy Optimization](https://arxiv.org/abs/2506.21655) [[Code](https://github.com/Indolent-Kawhi/View-R1)]

* [2506] [MMSearch-R1] [MMSearch-R1: Incentivizing LMMs to Search](https://arxiv.org/abs/2506.20670)  [[Code](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)]

* [2506] [PeRL] [PeRL: Permutation-Enhanced Reinforcement Learning for Interleaved Vision-Language Reasoning](https://arxiv.org/abs/2506.14907) [[Code](https://github.com/alchemistyzz/PeRL)]

* [2506] [MM-R5] [MM-R5: MultiModal Reasoning-Enhanced ReRanker via Reinforcement Learning for Document Retrieval](https://arxiv.org/abs/2506.12364) [[Model](https://huggingface.co/i2vec/MM-R5)]  [[Code](https://github.com/i2vec/MM-R5)]

* [2506] [ViCrit] [ViCrit: A Verifiable Reinforcement Learning Proxy Task for Visual Perception in VLMs](https://arxiv.org/abs/2506.10128) [[Models](https://huggingface.co/collections/russwang/vicrit-68489e13f223c00a6b6d5732)]  [[Datasets](https://huggingface.co/collections/russwang/vicrit-68489e13f223c00a6b6d5732)]  [[Code](https://github.com/si0wang/ViCrit)]

* [2506] [ViLaSR] [Reinforcing Spatial Reasoning in Vision-Language Models with Interwoven Thinking and Visual Drawing](https://arxiv.org/abs/2506.09965) [[Models](https://huggingface.co/collections/AntResearchNLP/vilasr-684a6ebbbbabe96eb77bbd6e)]  [[Datasets](https://huggingface.co/collections/AntResearchNLP/vilasr-684a6ebbbbabe96eb77bbd6e)]  [[Code](https://github.com/AntResearchNLP/ViLaSR)]

* [2506] [Vision Matters] [Vision Matters: Simple Visual Perturbations Can Boost Multimodal Math Reasoning](https://arxiv.org/abs/2506.09736) [[Model](https://huggingface.co/Yuting6/Vision-Matters-7B)]  [[Datasets](https://huggingface.co/collections/Yuting6/vision-matters-684801dd1879d3e639a930d1)]  [[Code](https://github.com/YutingLi0606/Vision-Matters)]

* [2506] [ViGaL] [Play to Generalize: Learning to Reason Through Game Play](https://arxiv.org/abs/2506.08011)  [[Project](https://yunfeixie233.github.io/ViGaL/)]  [[Model](https://huggingface.co/yunfeixie/ViGaL-7B)] [[Code](https://github.com/yunfeixie233/ViGaL)]

* [2506] [RAP] [Truth in the Few: High-Value Data Selection for Efficient Multi-Modal Reasoning](https://arxiv.org/abs/2506.04755)  [[Code](https://github.com/Leo-ssl/RAP)]

* [2506] [RACRO] [Perceptual Decoupling for Scalable Multi-modal Reasoning via Reward-Optimized Captioning](https://arxiv.org/abs/2506.04559)  [[Models](https://huggingface.co/collections/KaiChen1998/racro-6848ec8c65b3a0bf33d0fbdb)] [[Demo](https://huggingface.co/spaces/Emova-ollm/RACRO-demo)] [[Code](https://github.com/gyhdog99/RACRO2/)]

* [2506] [Revisual-R1] [Advancing Multimodal Reasoning: From Optimized Cold Start to Staged Reinforcement Learning](https://arxiv.org/abs/2506.04207)  [[Models](https://huggingface.co/collections/csfufu/revisual-r1-6841b748f08ee6780720c00e)]  [[Code](https://github.com/CSfufu/Revisual-R1)]

* [2506] [Rex-Thinker] [Rex-Thinker: Grounded Object Referring via Chain-of-Thought Reasoning](https://arxiv.org/abs/2506.04034)  [[Project](https://rexthinker.github.io/)]  [[Model](https://huggingface.co/IDEA-Research/Rex-Thinker-GRPO-7B)]  [[Dataset](https://huggingface.co/datasets/IDEA-Research/HumanRef-CoT-45k)]  [[Demo](https://huggingface.co/spaces/Mountchicken/Rex-Thinker-Demo)]  [[Code](https://github.com/IDEA-Research/Rex-Thinker)]

* [2506] [ControlThinker (generation)] [ControlThinker: Unveiling Latent Semantics for Controllable Image Generation through Visual Reasoning](https://arxiv.org/abs/2506.03596)  [[Code](https://github.com/Maplebb/ControlThinker)]

* [2506] [Multimodal DeepResearcher] [Multimodal DeepResearcher: Generating Text-Chart Interleaved Reports From Scratch with Agentic Framework](https://arxiv.org/abs/2506.02454)  [[Project](https://rickyang1114.github.io/multimodal-deepresearcher/)]

* [2506] [SynthRL] [SynthRL: Scaling Visual Reasoning with Verifiable Data Synthesis](https://arxiv.org/abs/2506.02096)  [[Model](https://huggingface.co/Jakumetsu/SynthRL-A-MMK12-8K-7B)]  [[Datasets](https://huggingface.co/collections/Jakumetsu/synthrl-6839d265136fa9ca717105c5)]  [[Code](https://github.com/NUS-TRAIL/SynthRL)]

* [2506] [SRPO] [SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning](https://arxiv.org/abs/2506.01713)  [[Project](https://srpo.pages.dev/)]  [[Dataset](https://huggingface.co/datasets/SRPOMLLMs/srpo-sft-data)]  [[Code](https://github.com/SUSTechBruce/SRPO_MLLMs)]

* [2506] [GThinker] [GThinker: Towards General Multimodal Reasoning via Cue-Guided Rethinking](https://arxiv.org/abs/2506.01078)  [[Model](https://huggingface.co/JefferyZhan/GThinker-7B)]  [[Datasets](https://huggingface.co/collections/JefferyZhan/gthinker-683e920eff706ead8fde3fc0)]  [[Code](https://github.com/jefferyZhan/GThinker)]

* [2505] [ReasonGen-R1 (generation)] [ReasonGen-R1: CoT for Autoregressive Image generation models through SFT and RL](https://arxiv.org/abs/2505.24875)  [[Project](https://reasongen-r1.github.io/)]  [[Models](https://huggingface.co/collections/Franklin0/reasongen-r1-6836ed61fc4f6db543c0d368)]  [[Datasets](https://huggingface.co/collections/Franklin0/reasongen-r1-6836ed61fc4f6db543c0d368)]  [[Code](https://github.com/Franklin-Zhang0/ReasonGen-R1)]

* [2505] [MoDoMoDo] [MoDoMoDo: Multi-Domain Data Mixtures for Multimodal LLM Reinforcement Learning](https://arxiv.org/abs/2505.24871) [[Project](https://modomodo-rl.github.io/)] [[Datasets](https://huggingface.co/yiqingliang)]  [[Code](https://github.com/lynl7130/MoDoMoDo)]

* [2505] [DINO-R1] [DINO-R1: Incentivizing Reasoning Capability in Vision Foundation Models](https://arxiv.org/abs/2505.24025)  [[Project](https://christinepan881.github.io/DINO-R1/)]  

* [2505] [VisualSphinx] [VisualSphinx: Large-Scale Synthetic Vision Logic Puzzles for RL](https://arxiv.org/abs/2505.23977)  [[Project](https://visualsphinx.github.io/)]  [[Model](https://huggingface.co/VisualSphinx/VisualSphinx-Difficulty-Tagging)]  [[Datasets](https://huggingface.co/collections/VisualSphinx/visualsphinx-v1-6837658bb93aa1e23aef1c3f)]  [[Code](https://github.com/VisualSphinx/VisualSphinx)]

* [2505] [PixelThink] [PixelThink: Towards Efficient Chain-of-Pixel Reasoning](https://arxiv.org/abs/2505.23727)  [[Project](https://pixelthink.github.io/)]  [[Code](https://github.com/songw-zju/PixelThink)]

* [2505] [ViGoRL] [Grounded Reinforcement Learning for Visual Reasoning](https://arxiv.org/abs/2505.23678)  [[Project](https://visually-grounded-rl.github.io/)]  [[Code](https://github.com/Gabesarch/grounded-rl)]

* [2505] [Jigsaw-R1] [Jigsaw-R1: A Study of Rule-based Visual Reinforcement Learning with Jigsaw Puzzles](https://arxiv.org/abs/2505.23590) [[Datasets](https://huggingface.co/jigsaw-r1)]   [[Code](https://github.com/zifuwanggg/Jigsaw-R1)]

* [2505] [UniRL] [UniRL: Self-Improving Unified Multimodal Models via Supervised and Reinforcement Learning](https://arxiv.org/abs/2505.23380) [[Model](https://huggingface.co/benzweijia/UniRL)]   [[Code](https://github.com/showlab/UniRL)]

* [2505] [Infi-MMR] [Infi-MMR: Curriculum-based Unlocking Multimodal Reasoning via Phased Reinforcement Learning in Multimodal Small Language Models](https://arxiv.org/abs/2505.23091) [[Model](https://huggingface.co/InfiX-ai/Infi-MMR-3B)] [[Code](https://github.com/InfiXAI/Infi-MMR)]

* [2505] [cadrille (generation)] [cadrille: Multi-modal CAD Reconstruction with Online Reinforcement Learning](https://arxiv.org/abs/2505.22914) 

* [2505] [SAM-R1] [SAM-R1: Leveraging SAM for Reward Feedback in Multimodal Segmentation via Reinforcement Learning](https://arxiv.org/abs/2505.22596) 

* [2505] [Thinking with Generated Images] [Thinking with Generated Images](https://arxiv.org/abs/2505.22525) [[Models](https://huggingface.co/GAIR/twgi-subgoal-anole-7b)]  [[Code](https://github.com/GAIR-NLP/thinking-with-generated-images)]

* [2505] [MM-UPT] [Unsupervised Post-Training for Multi-Modal LLM Reasoning via GRPO](https://arxiv.org/abs/2505.22453) [[Model](https://huggingface.co/WaltonFuture/Qwen2.5-VL-7B-MM-UPT-MMR1)]  [[Dataset](https://huggingface.co/datasets/WaltonFuture/MMR1-direct-synthesizing)]  [[Code](https://github.com/waltonfuture/MM-UPT)]

* [2505] [RL-with-Cold-Start] [Advancing Multimodal Reasoning via Reinforcement Learning with Cold Start](https://arxiv.org/abs/2505.22334) [[Models](https://huggingface.co/WaltonFuture/Qwen2.5VL-7b-RLCS)]  [[Datasets](https://huggingface.co/datasets/WaltonFuture/Multimodal-Cold-Start)]  [[Code](https://github.com/waltonfuture/RL-with-Cold-Start)]

* [2505] [VRAG-RL] [VRAG-RL: Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning with Reinforcement Learning](https://arxiv.org/abs/2505.22019) [[Models](https://huggingface.co/autumncc/Qwen2.5-VL-7B-VRAG)]  [[Code](https://github.com/Alibaba-NLP/VRAG)]

* [2505] [MLRM-Halu] [More Thinking, Less Seeing? Assessing Amplified Hallucination in Multimodal Reasoning Models](https://arxiv.org/abs/2505.21523) [[Project](https://mlrm-halu.github.io/)] [[Benchmark](https://huggingface.co/datasets/LCZZZZ/RH-Bench)]  [[Code](https://github.com/MLRM-Halu/MLRM-Halu)]

* [2505] [Active-O3] [Active-O3: Empowering Multimodal Large Language Models with Active Perception via GRPO](https://arxiv.org/abs/2505.21457) [[Project](https://aim-uofa.github.io/ACTIVE-o3/)] [[Model](https://www.modelscope.cn/models/zzzmmz/ACTIVE-o3)]  [[Code](https://github.com/aim-uofa/Active-o3)]

### Mathematical and Spatial Reasoning
- **MAYE: Rethinking RL Scaling for Vision Language Models** - [arXiv:2504.02587](https://arxiv.org/abs/2504.02587) *(arXiv 2024)*  
  *Transparent framework for visual mathematical reasoning with RL*

- **Vision Matters: Simple Visual Perturbations Can Boost Multimodal Math Reasoning** - [arXiv:2506.09736](https://arxiv.org/abs/2506.09736) *(arXiv 2025)*  
  *Visual perturbations and RL significantly improve mathematical reasoning*

### Tool Use and Agentic Applications
- **Reinforcing VLMs to Use Tools for Detailed Visual Reasoning** - [arXiv:2506.14821](https://arxiv.org/abs/2506.14821) *(arXiv 2025)*  
  *GRPO training for zoom tool usage in resource-constrained settings*

- **VisionThink: Smart and Efficient VLM via Reinforcement Learning** - [arXiv:2507.13348](https://arxiv.org/abs/2507.13348) *(arXiv 2025)*  
  *Dynamic visual token compression with autonomous resolution decisions*

### Medical and Domain-Specific Applications
- **MedVLM-R1: Incentivizing Medical Reasoning Capability via RL** - [arXiv:2502.19634](https://arxiv.org/abs/2502.19634) *(arXiv 2025)*  
  *Medical imaging analysis and clinical reasoning enhancement*

- **Patho-R1: A Multimodal RL-Based Pathology Expert Reasoner** - [arXiv:2505.11404](https://arxiv.org/abs/2505.11404) *(arXiv 2025)*  
  *Specialized pathology reasoning for histopathology analysis*

### Video and Temporal Reasoning
- **Video-R1: Reinforcing Video Reasoning in MLLMs** - [arXiv:2503.21776](https://arxiv.org/abs/2503.21776) *(arXiv 2025)*  
  *R1-style reasoning extended to video understanding and temporal reasoning*

- **VideoChat-R1: Enhancing Spatio-Temporal Perception via RL Fine-Tuning** - [arXiv:2504.06958](https://arxiv.org/abs/2504.06958) *(arXiv 2025)*  
  *Improved spatio-temporal perception in video understanding*


### Embodied Vision

* [2508] [Embodied-R1] [Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation](https://arxiv.org/abs/2508.13998) [[Project](https://embodied-r1.github.io/)]  [[Models](https://huggingface.co/collections/IffYuan/embodied-r1-684a8474b3a49210995f9081)]  [[Datasets](https://huggingface.co/collections/IffYuan/embodied-r1-684a8474b3a49210995f9081)]  [[Code](https://github.com/pickxiguapi/Embodied-R1)]

* [2508] [Affordance-R1] [Affordance-R1: Reinforcement Learning for Generalizable Affordance Reasoning in Multimodal Large Language Model](https://arxiv.org/abs/2508.06206) [[Model](https://huggingface.co/hqking/affordance-r1)]  [[Code](https://github.com/hq-King/Affordance-R1)]

* [2508] [VL-DAC] [Enhancing Vision-Language Model Training with Reinforcement Learning in Synthetic Worlds for Real-World Success](https://arxiv.org/abs/2508.04280)  [[Code](https://github.com/corl-team/VL-DAC)] 

* [2507] [ThinkAct] [ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning](https://arxiv.org/abs/2507.16815) [[Project](https://jasper0314-huang.github.io/thinkact-vla/)] 

* [2506] [VLN-R1] [VLN-R1: Vision-Language Navigation via Reinforcement Fine-Tuning](https://arxiv.org/abs/2506.17221) [[Project](https://vlnr1.github.io/)] [[Dataset](https://huggingface.co/datasets/alexzyqi/VLN-Ego)]  [[Code](https://github.com/Qi-Zhangyang/GPT4Scene-and-VLN-R1)]

* [2506] [VIKI-R] [VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning](https://arxiv.org/abs/2506.09049) [[Project](https://faceong.github.io/VIKI-R/)] [[Dataset](https://huggingface.co/datasets/henggg/VIKI-R)]  [[Code](https://github.com/MARS-EAI/VIKI-R)]

* [2506] [RoboRefer] [RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics](https://arxiv.org/abs/2506.04308) [[Project](https://zhoues.github.io/RoboRefer/)] [[Dataset](https://huggingface.co/datasets/BAAI/RefSpatial-Bench)]  [[Code](https://github.com/Zhoues/RoboRefer)]

* [2506] [Robot-R1] [Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics](https://arxiv.org/abs/2506.00070) 

* [2505] [VLA RL Study] [What Can RL Bring to VLA Generalization? An Empirical Study](https://arxiv.org/abs/2505.19789) [[Project](https://rlvla.github.io/)]  [[Models](https://huggingface.co/collections/gen-robot/rlvla-684bc48aa6cf28bac37c57a2)] [[Code](https://github.com/gen-robot/RL4VLA)]

* [2505] [VLA-RL] [VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning](https://arxiv.org/abs/2505.18719) [[Code](https://github.com/GuanxingLu/vlarl)]

* [2505] [ManipLVM-R1] [ManipLVM-R1: Reinforcement Learning for Reasoning in Embodied Manipulation with Large Vision-Language Models](https://arxiv.org/abs/2505.16517) 

* [2504] [Embodied-R] [Embodied-R: Collaborative Framework for Activating Embodied Spatial Reasoning in Foundation Models via Reinforcement Learning](https://arxiv.org/abs/2504.12680) [[Code](https://github.com/EmbodiedCity/Embodied-R.code)]

* [2503] [Embodied-Reasoner] [Embodied-Reasoner: Synergizing Visual Search, Reasoning, and Action for Embodied Interactive Tasks](https://arxiv.org/abs/2503.21696v1) [[Project](https://embodied-reasoner.github.io/)] [[Dataset](https://huggingface.co/datasets/zwq2018/embodied_reasoner)] [[Code](https://github.com/zwq2018/embodied_reasoner)]


### Autonomous Driving

* [2507] [DriveAgent-R1] [DriveAgent-R1: Advancing VLM-based Autonomous Driving with Hybrid Thinking and Active Perception](https://arxiv.org/abs/2507.20879) 

* [2506] [Drive-R1] [Drive-R1: Bridging Reasoning and Planning in VLMs for Autonomous Driving with Reinforcement Learning](https://arxiv.org/abs/2506.18234) 

* [2506] [AutoVLA] [AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](https://arxiv.org/abs/2506.13757) [[Project 🌐](https://autovla.github.io/)]  [[Code 💻](https://github.com/ucla-mobility/AutoVLA)]

* [2505] [AgentThink] [AgentThink: A Unified Framework for Tool-Augmented Chain-of-Thought Reasoning in Vision-Language Models for Autonomous Driving](https://arxiv.org/abs/2505.15298)

---

## Benchmarks and Evaluation

### Mathematical Reasoning
- **MATH Dataset** - Competition-level mathematics problems
- **GSM8K** - Grade school math word problems  
- **AIME** - American Invitational Mathematics Examination
- **AMC** - American Mathematics Competitions

### Code Generation  
- **HumanEval** - Python programming problems
- **MBPP** - Mostly Basic Python Programming
- **CodeContests** - Programming competition problems

### Multimodal Reasoning
- **MathVista** - Mathematical reasoning with visual elements
- **ScienceQA** - Science question answering with diagrams
- **VQA** - Visual Question Answering benchmarks

### Specialized Evaluation
- **RewardBench** - First comprehensive reward model benchmark
- **LOCOMO** - Long-context memory evaluation (600-turn dialogues)
- **WebArena-Lite** - Web navigation task evaluation

---


* [2508] [MM-BrowseComp] [MM-BrowseComp: A Comprehensive Benchmark for Multimodal Browsing Agents](https://arxiv.org/abs/2508.13186)  [[Code](https://github.com/MMBrowseComp/MM-BrowseComp)]

* [2508] [HumanSense] [HumanSense: From Multimodal Perception to Empathetic Context-Aware Responses through Reasoning MLLMs](https://arxiv.org/abs/2508.10576)  [[Project](https://digital-avatar.github.io/ai/HumanSense/)] 

* [2508] [MathReal] [MathReal: We Keep It Real! A Real Scene Benchmark for Evaluating Math Reasoning in Multimodal Large Language Models](https://arxiv.org/abs/2508.06009)  [[Dataset](https://huggingface.co/datasets/junfeng0288/MathReal)]  [[Code](https://github.com/junfeng0288/MathReal)]

* [2508] [DeepPHY] [DeepPHY: Benchmarking Agentic VLMs on Physical Reasoning](https://arxiv.org/abs/2508.05405)  [[Code](https://github.com/XinrunXu/DeepPHY)] 

* [2507] [Zebra-CoT] [Zebra-CoT: A Dataset for Interleaved Vision Language Reasoning](https://arxiv.org/abs/2507.16746)  [[Dataset](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT)]  [[Code](https://github.com/multimodal-reasoning-lab/Bagel-Zebra-CoT)]

* [2507] [Video-TT] [Towards Video Thinking Test: A Holistic Benchmark for Advanced Video Reasoning and Understanding](https://arxiv.org/abs/2507.15028) [[Project](https://zhangyuanhan-ai.github.io/video-tt/)]  [[Dataset](https://huggingface.co/datasets/lmms-lab/video-tt)] 

* [2507] [EmbRACE-3K] [EmbRACE-3K: Embodied Reasoning and Action in Complex Environments](https://arxiv.org/abs/2507.10548) [[Project](https://mxllc.github.io/EmbRACE-3K/)] [[Code](https://github.com/mxllc/EmbRACE-3K)]

* [2506] [PhysUniBench] [PhysUniBench: An Undergraduate-Level Physics Reasoning Benchmark for Multimodal Models](https://arxiv.org/abs/2506.17667) [[Project](https://prismax-team.github.io/PhysUniBenchmark/)]  [[Dataset](https://huggingface.co/datasets/PrismaX/PhysUniBench)]  [[Code](https://github.com/PrismaX-Team/PhysUniBenchmark)]

* [2506] [MMReason] [MMReason: An Open-Ended Multi-Modal Multi-Step Reasoning Benchmark for MLLMs Toward AGI](https://arxiv.org/abs/2506.23563)  [[Code](https://github.com/HJYao00/MMReason)]

* [2506] [MindCube] [Spatial Mental Modeling from Limited Views](https://arxiv.org/abs/2506.21458) [[Project](https://mind-cube.github.io/)]  [[Models](https://huggingface.co/MLL-Lab/models)]  [[Dataset](https://huggingface.co/datasets/MLL-Lab/MindCube)] [[Code](https://github.com/mll-lab-nu/MindCube)]

* [2506] [VRBench] [VRBench: A Benchmark for Multi-Step Reasoning in Long Narrative Videos](https://arxiv.org/abs/2506.10857) [[Project](https://vrbench.github.io/)] [[Dataset](https://huggingface.co/datasets/OpenGVLab/VRBench)] [[Code](https://github.com/OpenGVLab/VRBench)]

* [2506] [MORSE-500] [MORSE-500: A Programmatically Controllable Video Benchmark to Stress-Test Multimodal Reasoning](https://arxiv.org/abs/2506.05523) [[Project](https://morse-500.github.io/)] [[Dataset](https://huggingface.co/datasets/video-reasoning/morse-500)] [[Code](https://github.com/morse-benchmark/morse-500)]

* [2506] [VideoMathQA] [VideoMathQA: Benchmarking Mathematical Reasoning via Multimodal Understanding in Videos](https://arxiv.org/abs/2506.05349) [[Project](https://mbzuai-oryx.github.io/VideoMathQA/)] [[Dataset](https://huggingface.co/datasets/MBZUAI/VideoMathQA)] [[Code](https://github.com/mbzuai-oryx/VideoMathQA)]

* [2506] [MMRB] [Evaluating MLLMs with Multimodal Multi-image Reasoning Benchmark](https://arxiv.org/abs/2506.04280) [[Project](https://mmrb-benchmark.github.io/)] [[Dataset](https://huggingface.co/datasets/HarrytheOrange/MMRB)] [[Code](https://github.com/LesterGong/MMRB)]

* [2506] [MMR-V] [MMR-V: What's Left Unsaid? A Benchmark for Multimodal Deep Reasoning in Videos](https://arxiv.org/abs/2506.04141) [[Project](https://mmr-v.github.io/)] [[Dataset](https://huggingface.co/datasets/JokerJan/MMR-VBench)] [[Code](https://github.com/GaryStack/MMR-V)]

* [2506] [OmniSpatial] [OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](https://arxiv.org/abs/2506.03135) [[Project](https://qizekun.github.io/omnispatial/)] [[Dataset](https://huggingface.co/datasets/qizekun/OmniSpatial)] [[Code](https://github.com/qizekun/OmniSpatial)]

* [2506] [VS-Bench] [VS-Bench: Evaluating VLMs for Strategic Reasoning and Decision-Making in Multi-Agent Environments](https://arxiv.org/abs/2506.02387) [[Project](https://vs-bench.github.io/)] [[Dataset](https://huggingface.co/datasets/zelaix/VS-Bench)] [[Code](https://github.com/zelaix/VS-Bench)]

* [2505] [Open CaptchaWorld] [Open CaptchaWorld: A Comprehensive Web-based Platform for Testing and Benchmarking Multimodal LLM Agents](https://arxiv.org/abs/2505.24878)  [[Dataset](https://huggingface.co/datasets/YaxinLuo/Open_CaptchaWorld)] [[Code](https://github.com/MetaAgentX/OpenCaptchaWorld)]

* [2505] [FinMME] [FinMME: Benchmark Dataset for Financial Multi-Modal Reasoning Evaluation](https://arxiv.org/abs/2505.24714)  [[Dataset](https://huggingface.co/datasets/luojunyu/FinMME)] [[Code](https://github.com/luo-junyu/FinMME)]

* [2505] [CSVQA] [CSVQA: A Chinese Multimodal Benchmark for Evaluating STEM Reasoning Capabilities of VLMs](https://arxiv.org/abs/2505.24120)  [[Dataset](https://huggingface.co/datasets/Skywork/CSVQA)] [[Code](https://github.com/SkyworkAI/CSVQA)]

* [2505] [VideoReasonBench] [VideoReasonBench: Can MLLMs Perform Vision-Centric Complex Video Reasoning?](https://arxiv.org/abs/2505.23359) [[Project](https://llyx97.github.io/video_reason_bench/)] [[Dataset](https://huggingface.co/datasets/lyx97/reasoning_videos)] [[Code](https://github.com/llyx97/video_reason_bench)]

* [2505] [Video-Holmes] [Video-Holmes: Can MLLM Think Like Holmes for Complex Video Reasoning?](https://arxiv.org/abs/2505.21374) [[Project](https://video-holmes.github.io/Page.github.io/)] [[Dataset](https://huggingface.co/datasets/TencentARC/Video-Holmes)] [[Code](https://github.com/TencentARC/Video-Holmes)]

* [2505] [MME-Reasoning] [MME-Reasoning: A Comprehensive Benchmark for Logical Reasoning in MLLMs](https://arxiv.org/abs/2505.21327) [[Project](https://alpha-innovator.github.io/mmereasoning.github.io/)] [[Dataset](https://huggingface.co/datasets/U4R/MME-Reasoning)] [[Code](https://github.com/Alpha-Innovator/MME-Reasoning)]

* [2505] [MMPerspective] [MMPerspective: Do MLLMs Understand Perspective? A Comprehensive Benchmark for Perspective Perception, Reasoning, and Robustness](https://arxiv.org/abs/2505.20426) [[Project](https://yunlong10.github.io/MMPerspective/)] [[Code](https://github.com/yunlong10/MMPerspective)]

* [2505] [SeePhys] [SeePhys: Does Seeing Help Thinking? -- Benchmarking Vision-Based Physics Reasoning](https://arxiv.org/abs/2505.19099) [[Project](https://seephys.github.io/)] [[Dataset](https://huggingface.co/datasets/SeePhys/SeePhys)] [[Code](https://github.com/SeePhys/seephys-project)] 

* [2505] [CXReasonBench] [CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays](https://arxiv.org/abs/2505.18087)  [[Code](https://github.com/ttumyche/CXReasonBench)] 

* [2505] [OCR-Reasoning] [OCR-Reasoning Benchmark: Unveiling the True Capabilities of MLLMs in Complex Text-Rich Image Reasoning](https://arxiv.org/abs/2505.17163) [[Project](https://ocr-reasoning.github.io/)] [[Dataset](https://huggingface.co/datasets/mx262/OCR-Reasoning)] [[Code](https://github.com/SCUT-DLVCLab/OCR-Reasoning)] 

* [2505] [RBench-V] [RBench-V: A Primary Assessment for Visual Reasoning Models with Multi-modal Outputs](https://arxiv.org/abs/2505.16770) [[Project](https://evalmodels.github.io/rbenchv/)] [[Dataset](https://huggingface.co/datasets/R-Bench/R-Bench-V)] [[Code](https://github.com/CHEN-Xinsheng/VLMEvalKit_RBench-V)] 

* [2505] [MMMR] [MMMR: Benchmarking Massive Multi-Modal Reasoning Tasks](https://arxiv.org/abs/2505.16459) [[Project](https://mmmr-benchmark.github.io)] [[Dataset](https://huggingface.co/datasets/csegirl/MMMR)] [[Code](https://github.com/CsEgir/MMMR)]

* [2505] [ReasonMap] [Can MLLMs Guide Me Home? A Benchmark Study on Fine-Grained Visual Reasoning from Transit Maps](https://arxiv.org/abs/2505.18675) [[Project](https://fscdc.github.io/Reason-Map/)] [[Dataset](https://huggingface.co/datasets/FSCCS/ReasonMap)] [[Code](https://github.com/fscdc/ReasonMap)] 

* [2505] [PhyX] [PhyX: Does Your Model Have the "Wits" for Physical Reasoning?](https://arxiv.org/abs/2505.15929) [[Project](https://phyx-bench.github.io/)] [[Dataset](https://huggingface.co/datasets/Cloudriver/PhyX)] [[Code](https://github.com/NastyMarcus/PhyX)] 

* [2505] [NOVA] [NOVA: A Benchmark for Anomaly Localization and Clinical Reasoning in Brain MRI](https://arxiv.org/abs/2505.14064) 


---

## Open-Source Projects

### Training Framework

* **TRL (Transformers Reinforcement Learning)** - [Hugging Face Documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer)  
  *Official Hugging Face implementation of GRPO with comprehensive trainer*

* **VERL (Versatile Efficient Reinforcement Learning)** - ByteDance  
  *Implementation with vLLM integration and SGLang support*

* **Unsloth GRPO Implementation** - [Blog Post](https://unsloth.ai/blog/r1-reasoning)  
  *Memory-efficient implementation reducing VRAM usage by 80%*

* [EasyR1](https://github.com/hiyouga/EasyR1)  ![EasyR1](https://img.shields.io/github/stars/hiyouga/EasyR1) (An Efficient, Scalable, Multi-Modality RL Training Framework)

* ms-swift

* llama-factory

### Vision (Image)

* [R1-V](https://github.com/Deep-Agent/R1-V)  ![R1-V](https://img.shields.io/github/stars/Deep-Agent/R1-V) [Blog](https://deepagent.notion.site/rlvr-in-vlms) [Datasets](https://huggingface.co/collections/MMInstruction/r1-v-67aae24fa56af9d2e2755f82)

* [Multimodal Open R1](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal)  ![Multimodal Open R1](https://img.shields.io/github/stars/EvolvingLMMs-Lab/open-r1-multimodal) [Model](https://huggingface.co/lmms-lab/Qwen2-VL-2B-GRPO-8k) [Dataset](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified)

* [MMR1](https://github.com/LengSicong/MMR1) ![LengSicong/MMR1](https://img.shields.io/github/stars/LengSicong/MMR1) [Code](https://github.com/LengSicong/MMR1) [Model](https://huggingface.co/MMR1/MMR1-Math-v0-7B) [Dataset](https://huggingface.co/datasets/MMR1/MMR1-Math-RL-Data-v0) 

* [R1-Multimodal-Journey](https://github.com/FanqingM/R1-Multimodal-Journey) ![R1-Multimodal-Journey](https://img.shields.io/github/stars/FanqingM/R1-Multimodal-Journey) (Latest progress at [MM-Eureka](https://github.com/ModalMinds/MM-EUREKA))

* [R1-Vision](https://github.com/yuyq96/R1-Vision) ![R1-Vision](https://img.shields.io/github/stars/yuyq96/R1-Vision) [Cold-Start Datasets](https://huggingface.co/collections/yuyq96/r1-vision-67a6fb7898423dca453efa83)

* [Ocean-R1](https://github.com/VLM-RL/Ocean-R1)  ![Ocean-R1](https://img.shields.io/github/stars/VLM-RL/Ocean-R1) [Models](https://huggingface.co/minglingfeng) [Datasets](https://huggingface.co/minglingfeng)

* [R1V-Free](https://github.com/Exgc/R1V-Free)  ![Exgc/R1V-Free](https://img.shields.io/github/stars/Exgc/R1V-Free) [Models](https://huggingface.co/collections/Exgc/r1v-free-67f769feedffab8761b8f053) [Dataset](https://huggingface.co/datasets/Exgc/R1V-Free_RLHFV)

* [SeekWorld](https://github.com/TheEighthDay/SeekWorld)  ![TheEighthDay/SeekWorld](https://img.shields.io/github/stars/TheEighthDay/SeekWorld) [Model](https://huggingface.co/TheEighthDay/SeekWorld_RL_PLUS) [Dataset](https://huggingface.co/datasets/TheEighthDay/SeekWorld) [Demo](https://huggingface.co/spaces/TheEighthDay/SeekWorld_APP)

* [R1-Track](https://github.com/Wangbiao2/R1-Track)  ![Wangbiao2/R1-Track](https://img.shields.io/github/stars/Wangbiao2/R1-Track) [Models](https://huggingface.co/WangBiao) [Datasets](https://huggingface.co/WangBiao)

### Vision (Video)

* [Open R1 Video](https://github.com/Wang-Xiaodong1899/Open-R1-Video) ![Open R1 Video](https://img.shields.io/github/stars/Wang-Xiaodong1899/Open-R1-Video) [Models](https://huggingface.co/Xiaodong/Open-R1-Video-7B)  [Datasets](https://huggingface.co/datasets/Xiaodong/open-r1-video-4k)

* [Open-LLaVA-Video-R1](https://github.com/Hui-design/Open-LLaVA-Video-R1) ![Open-LLaVA-Video-R1](https://img.shields.io/github/stars/Hui-design/Open-LLaVA-Video-R1) [Code](https://github.com/Hui-design/Open-LLaVA-Video-R1)

### Agent

* [VAGEN](https://github.com/RAGEN-AI/VAGEN) ![VAGEN](https://img.shields.io/github/stars/RAGEN-AI/VAGEN) [Code](https://github.com/RAGEN-AI/VAGEN)



## Citation

If you find this collection useful, please cite:

```bibtex
@misc{awesome-rl-reasoning-lms-2025,
  title={Awesome RL for Reasoning in Large Language Models},
  author={Jiaqi Liu},
  year={2025},
  url={https://github.com/**/Awesome-RL-Reasoning-LMs}
}
```
