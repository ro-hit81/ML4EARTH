# 🌍 Prithvi4QR: Foundation Models for High-Resolution Earth Observation Mapping

[![ML4Earth](https://img.shields.io/badge/ML4Earth-2024_Winner-gold?style=for-the-badge)](https://ml4earth24.devpost.com/)
[![Prithvi](https://img.shields.io/badge/Foundation_Model-Prithvi_ViT_100-blue?style=for-the-badge)](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M)
[![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![TensorBoard](https://img.shields.io/badge/TensorBoard-Metrics-orange?style=for-the-badge&logo=tensorflow)](https://tensorboard.dev/)

<div align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">
</div>

## 🏆 **Competition Achievement**
**Winner of ML4Earth - Foundation Models for Earth Observation Hackathon 2024**  
*September 18-23, 2024 | €1,800 in prizes | 139 participants*

---

## 🌍 **Project Overview**

**Prithvi4QR** is an advanced Earth Observation foundation model designed for quick-response (QR) teams to support decision makers during climate-related disasters. By fine-tuning the NASA-IBM Prithvi-ViT-100 foundation model, our solution provides rapid, accurate spatial information for effective resource allocation during large-scale disasters like floods, wildfires, and other climate emergencies.

### 🎯 **What it Does**
- **Input**: Optical images of size 512×512 pixels
- **Output**: High-resolution segmentation masks with 5 critical land cover classes
- **Purpose**: Supporting quick-response teams with automated spatial analysis
- **Impact**: Reducing manual interpretation time for emergency decision-making

---

## 🚀 **Key Features**

### 🧠 **Advanced Architecture**
- **Foundation Model**: NASA-IBM Prithvi-ViT-100 (100M parameters)
- **Decoder**: UperNet for semantic segmentation
- **Fine-tuned Parameters**: 12.1 million parameters
- **Training Strategy**: Frozen backbone with trainable decoder

### 🎨 **Land Cover Classification**
| **Class** | **Description** | **Use Case** |
|-----------|----------------|--------------|
| 🌫️ **Background** | Non-classified areas | General terrain |
| 🏢 **Buildings** | Urban structures | Infrastructure assessment |
| 🌲 **Woodland** | Forest areas | Environmental monitoring |
| 💧 **Water** | Water bodies | Flood extent mapping |
| 🛣️ **Roads** | Transportation networks | Accessibility analysis |

### ⚡ **Performance Highlights**
- **Training Duration**: 50 epochs maximum
- **Overall Accuracy**: 90.43% on test data
- **Input Resolution**: 512×512 pixels  
- **GPU Acceleration**: Single device training
- **Class Balancing**: Advanced weighting for imbalanced datasets

---

## 🔬 **Technical Implementation**

### 📊 **Model Architecture**
```python
model_args = {
    "backbone": "prithvi_vit_100",
    "decoder": "UperNetDecoder", 
    "in_channels": 3,
    "num_classes": 5,
    "bands": [HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE],
    "pretrained": True,
    "decoder_channels": 256,
    "head_dropout": 0.1,
    "freeze_backbone": True
}
```

### 🎛️ **Training Configuration**
```yaml
# Key Training Parameters
batch_size: 32
learning_rate: 5e-4
optimizer: AdamW
weight_decay: 0.05
max_epochs: 50
precision: 16-mixed
accelerator: gpu
```

### ⚖️ **Class Balancing Strategy**
```python
class_weights = [0.02, 0.65, 0.04, 0.1, 0.4]
# Background, Buildings, Woodland, Water, Roads
```

---

## 🛠️ **Installation & Setup**

### 📋 **Prerequisites**
- Python 3.10+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### 🔧 **Environment Setup**
```bash
# Clone the repository
git clone https://github.com/ro-hit81/ML4EARTH.git
cd ML4EARTH

# Create conda environment
conda env create -f environment.yml
conda activate terratorch

# Download LandCover.ai dataset (automatic via torchgeo)
```

### 📦 **Key Dependencies**
- `terratorch`: Foundation model framework
- `pytorch-lightning`: Training orchestration  
- `torchgeo`: Geospatial datasets and transforms
- `tensorboard`: Experiment tracking
- `timm`: Model architectures

---

## 🚀 **Quick Start**

### 1️⃣ **Model Training**
```bash
# Fine-tune Prithvi model on LandCover.ai dataset
jupyter notebook prithvi_finetuning.ipynb
```

### 2️⃣ **Model Evaluation**
```bash
# Generate predictions and visualizations
jupyter notebook prediction.ipynb
```

### 3️⃣ **Performance Analysis**
```bash
# Analyze metrics and create reports
jupyter notebook analysis.ipynb
```

### 4️⃣ **Configuration-based Training**
```bash
# Alternative: Use YAML configuration
python -m lightning.pytorch.cli fit --config landcoverai_config.yaml
```

---

## 📊 **Project Structure**

```
ML4EARTH/
│
├── 📄 README.md                           # Project documentation
├── 📓 prithvi_finetuning.ipynb           # Main training notebook
├── 📓 prediction.ipynb                   # Inference and visualization
├── 📓 analysis.ipynb                     # Performance analysis
├── ⚙️ landcoverai_config.yaml            # Training configuration
├── 🐍 environment.yml                    # Conda environment
│
├── 📂 predicted/                          # Model outputs
├── 📂 data/landcoverai/                  # Dataset (auto-downloaded)
├── 📂 models/logs/                       # Training logs
└── 🎥 ML4EARTH Hackathon Introduction.mp4 # Project demo
```

---

## 📊 **Performance Metrics**

### 🎯 **Model Evaluation Results**
- **Overall Accuracy**: 90.43%
- **F1 Score**: 90.43%
- **Test Loss**: 0.247

### 📊 **Class-specific Performance**
| **Class** | **Accuracy** | **Jaccard Index** |
|-----------|--------------|-------------------|
| **Woodland** 🌲 | 96.12% | 85.21% |
| **Water** 💧 | 95.32% | 78.85% |
| **Building** 🏢 | 92.03% | 43.35% |
| **Background** 🌫️ | 86.82% | 85.02% |
| **Road** 🛣️ | 79.74% | 41.00% |

### 🎯 **Key Insights**
- **Best performing classes**: Woodland and Water detection
- **Challenging classes**: Roads and Buildings (lower IoU despite good accuracy)
- **Overall strong performance**: 90%+ overall accuracy with room for improvement in segmentation precision

---

## 🌟 **Innovation Highlights**

### 💡 **What Inspired Us**
Climate change disasters like European floods require rapid spatial intelligence for effective resource allocation. Our solution bridges the gap between raw satellite imagery and actionable insights for emergency response teams.

### 🔬 **Technical Achievements**
1. **First-time EO Foundation Model Fine-tuning**: Successfully adapted cutting-edge models
2. **Class Imbalance Solutions**: Advanced weighting strategies for real-world datasets
3. **Real-time Capability**: Optimized for emergency response scenarios
4. **Multi-framework Integration**: Seamless combination of terratorch, PyTorch Lightning, and TorchGeo

### 🏆 **What We're Proud Of**
- Successfully fine-tuned two different Earth Observation foundation models
- Achieved notable results under strict time constraints
- Created a production-ready solution for disaster response
- Implemented comprehensive evaluation framework

---

## 🔄 **Methodology Comparison**

### 🆚 **Foundation Model Evaluation**
| **Model** | **Architecture** | **Implementation** | **Selection** |
|-----------|------------------|-------------------|---------------|
| **Prithvi-ViT-100** ✅ | Vision Transformer | TeraTorch framework | Selected |
| **Satlas Aerial SwinB** | Swin Transformer | Alternative considered | Evaluated |

**Decision**: Prithvi model selected based on framework compatibility and foundation model capabilities.

---

## 🎯 **Use Cases & Applications**

### 🚨 **Emergency Response**
- **Flood Mapping**: Rapid water extent assessment
- **Infrastructure Damage**: Building and road network analysis  
- **Evacuation Planning**: Accessibility route identification
- **Resource Allocation**: Priority area determination

### 🌍 **Environmental Monitoring**
- **Deforestation Detection**: Woodland area changes
- **Urban Expansion**: Building development tracking
- **Water Resource Management**: Surface water monitoring
- **Disaster Impact Assessment**: Before/after comparisons

### 🏙️ **Urban Planning**
- **Infrastructure Development**: Smart city planning
- **Transportation Networks**: Road connectivity analysis
- **Green Space Management**: Urban forest monitoring
- **Population Distribution**: Settlement pattern analysis

---

## 🔬 **Technical Deep Dive**

### 🧪 **Training Strategy**
```python
# Freeze backbone for stable fine-tuning
freeze_backbone: True

# Dynamic learning rate scheduling
lr: 5e-4

# Mixed precision for memory efficiency  
precision: "16-mixed"

# Class weights for imbalanced data
class_weights: [0.02, 0.55, 0.04, 0.14, 0.25]
```

### 📊 **Data Augmentation**
- **Geometric**: Rotation, flipping, scaling
- **Photometric**: Brightness, contrast adjustments
- **Spatial**: Random cropping and resizing
- **Domain-specific**: Atmospheric effects simulation

### 🔍 **Loss Function Optimization**
- **Primary**: Weighted Cross-Entropy Loss
- **Focus**: Class imbalance handling
- **Monitoring**: Jaccard Index for critical classes
- **Early Stopping**: Patience-based convergence

---

## 🚧 **Challenges & Solutions**

### ⚠️ **Technical Challenges**
1. **Model Integration**: Adapting foundation models to specific frameworks
2. **Class Imbalance**: Handling unequal distribution in land cover data
3. **Framework Compatibility**: Ensuring smooth integration with TeraTorch
4. **Evaluation Metrics**: Implementing comprehensive performance tracking

### ✅ **Solutions Implemented**
1. **Foundation Model Selection**: Chose Prithvi for framework compatibility
2. **Class Balancing**: Dynamic weight adjustment in loss function
3. **Training Optimization**: Mixed precision and efficient data loading
4. **Comprehensive Monitoring**: TensorBoard integration for metric tracking

---

## 🤝 **Contributing**

We welcome contributions to improve Prithvi4QR! Here's how you can help:

### 🐛 **Bug Reports**
- Use GitHub Issues for bug reports
- Include environment details and error logs
- Provide sample data and reproduction steps

### 💡 **Feature Requests**
- Additional land cover classes
- New foundation model backends
- Performance optimizations
- Deployment improvements

### 📝 **Development Guidelines**
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit pull request with documentation

---

## 📚 **References & Acknowledgments**

### 🛰️ **Foundation Models**
- **Prithvi**: [NASA-IBM Geospatial Foundation Model](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M)
- **Satlas**: [High-Resolution Satellite Imagery Models](https://github.com/allenai/satlas)

### 📖 **Key Publications**
- Clay et al. (2023): "Foundation Models for Earth Observation"
- NASA-IBM (2023): "Prithvi: A Foundation Model for Earth Observation"
- Bastani et al. (2023): "SatLas: A Large-Scale Multi-Modal Dataset"

### 🏆 **Competition**
- **ML4Earth Hackathon 2024**: [Official Page](https://ml4earth24.devpost.com/)
- **TUM Data Science in EO**: Competition organizers
- **GitHub Starter Pack**: [ML4Earth-Hackathon-2024](https://github.com/zhu-xlab/ML4Earth-Hackathon-2024)

### 🙏 **Libraries & Frameworks**
- **Terratorch**: Foundation model training framework
- **PyTorch Lightning**: Scalable deep learning
- **TorchGeo**: Geospatial machine learning datasets
- **TensorBoard**: Experiment tracking and visualization

---

## 📄 **License**

This project is open source and available under the [MIT License](LICENSE).

---

## 📞 **Contact**

### 👨‍💻 **Team Prithvi4QR**
- **GitHub**: [ro-hit81/ML4EARTH](https://github.com/ro-hit81/ML4EARTH)
- **Email**: [rhtkhati@gmail.com](mailto:rhtkhati@gmail.com)
- **Competition**: [ML4Earth DevPost](https://ml4earth24.devpost.com/)

### 🌟 **Project Links**
- **Demo Video**: [ML4EARTH Hackathon Introduction.mp4](./ML4EARTH%20Hackathon%20Introduction.mp4)
- **Live Demo**: Coming soon
- **Documentation**: This README and Jupyter notebooks

---

<div align="center">

### 🌍 **"Empowering emergency response through intelligent Earth observation."**

*Making satellite imagery analysis as fast as decision-making demands.*

![Visitor Count](https://komarev.com/ghpvc/?username=ro-hit81&label=Repository%20Views&color=brightgreen&style=flat-square)
[![GitHub stars](https://img.shields.io/github/stars/ro-hit81/ML4EARTH.svg?style=social&label=Star&maxAge=2592000)](https://github.com/ro-hit81/ML4EARTH/stargazers/)

</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png" width="100%">
</div>
