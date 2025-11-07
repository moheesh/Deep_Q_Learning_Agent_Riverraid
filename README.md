# Deep Q-Learning for Atari River Raid

A comprehensive implementation of Deep Q-Networks (DQN) with Dueling architecture for training an AI agent to play Atari River Raid. This project demonstrates advanced reinforcement learning techniques including GPU-optimized experience replay, systematic hyperparameter optimization, and novel exploration strategies.

## 🎮 Demo

https://youtu.be/UVknj2Ifwmg?si=epeigjrFahmCG5WH

**Performance Improvement: 30% over random baseline**
- Random Agent: 1,254 average score
- Trained Agent: 1,630 average score (2,210 peak)

## 📊 Key Features

- **Dueling DQN Architecture**: Separates value and advantage estimation for improved learning stability
- **60 FPS Training**: 4x faster than standard implementations through GPU optimization
- **Custom Environment Wrappers**: Reward shaping and frame preprocessing for optimal learning
- **Two-Phase Exploration**: Systematic action cycling followed by epsilon-greedy
- **GPU-Optimized Replay Buffer**: 75% memory reduction using uint8 storage
- **Hyperparameter Grid Search**: Automated optimization across multiple configurations

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA GPU with 8GB+ VRAM (RTX 2070 or better recommended)
- CUDA 11.8+
- 16GB RAM

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/dqn-river-raid.git
cd dqn-river-raid

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]
pip install ale-py opencv-python imageio imageio-ffmpeg tqdm matplotlib
```

### Training the Agent
```python
# Run the complete training pipeline
python train.py

# Or use Jupyter notebook
jupyter notebook DQN_River_Raid.ipynb
```

## 📁 Project Structure
```
dqn-river-raid/
│
├── DQN_River_Raid.ipynb    # Main implementation notebook
├── train.py                 # Standalone training script
├── requirements.txt         # Package dependencies
├── README.md               # This file
│
├── models/                 # Saved model checkpoints
│   └── final_model.pth
│
├── results/               # Training results and videos
│   ├── training_curves.png
│   └── gameplay_videos/
│
└── docs/                  # Additional documentation
    └── technical_report.pdf
```

## 🔧 Technical Implementation

### Network Architecture
```python
Dueling DQN Architecture:
Input: 84×84×4 (stacked frames)
├─ Conv1: 32 filters, 8×8, stride 4
├─ Conv2: 64 filters, 4×4, stride 2
├─ Conv3: 64 filters, 3×3, stride 1
├─ Flatten: 3136 features
├─ FC: 512 units
├─ Split:
│   ├─ Value Stream: 512→1
│   └─ Advantage Stream: 512→18
└─ Output: Q(s,a) = V(s) + (A(s,a) - mean(A))
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.00025 | RMSprop optimizer |
| Gamma | 0.99 | Discount factor |
| Epsilon Start | 1.0 | Initial exploration |
| Epsilon Final | 0.1 | Final exploration |
| Batch Size | 32 | Experience replay |
| Buffer Size | 50,000 | Replay buffer capacity |
| Target Update | 4,000 steps | Network sync frequency |

### Custom Components

#### EpisodicLifeWrapper
Applies -100 penalty for life loss, teaching self-preservation

#### FrameSkip
Handles Atari sprite flickering through max-pooling

#### Uniform Action Cycling
Systematic exploration ensuring complete action space coverage

## 📈 Results

### Training Progress
- Episodes 1-100: Exploration phase (800 avg score)
- Episodes 100-500: Rapid learning (1,150 avg score)
- Episodes 500-1,000: Strategy refinement (1,630 avg score)

### Performance Metrics
- Average Score Improvement: 30%
- Peak Score Improvement: 44%
- Episode Length Increase: 67%
- Training Time: ~3 hours on RTX 3080

## 🧪 Experiments

### Hyperparameter Search Results

| Configuration | LR | Gamma | Alpha | Final Score |
|--------------|-----|-------|-------|-------------|
| Baseline | 0.00025 | 0.99 | 0.95 | 1,630 |
| Aggressive | 0.0005 | 0.99 | 0.90 | 1,100 |
| Patient | 0.0001 | 0.98 | 0.95 | 980 |
| Explorer | 0.00025 | 0.99 | 0.95 | 1,580 |

## 📚 References

1. Mnih, V., et al. (2015). [Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236). Nature, 518(7540), 529-533.

2. Van Hasselt, H., et al. (2016). [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461). AAAI Conference on Artificial Intelligence.

3. Wang, Z., et al. (2016). [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581). ICML.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see below for details:
```
MIT License

Copyright (c) 2024 Moheesh Kavithaarumugam

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgments

- Gymnasium team for the Atari environment framework
- PyTorch team for the deep learning infrastructure
- Carol Shaw for creating the original River Raid game (Activision, 1982)

