<div align="center">

# 🚀 Deployment & Systems

### *Quantization, Pruning, ONNX, TensorRT*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/tutorials/blob/main/advanced_source/dynamic_quantization_tutorial.ipynb)

</div>

---

**Navigation:** [← Computational Photography](../17_Computational_Photography/) | [🏠 Home](../README.md) | [Ethics & Safety →](../19_Ethics_Safety/)

---

## 📖 Topics Covered
- Model Quantization
- Pruning
- Knowledge Distillation
- ONNX Export
- TensorRT Optimization
- Edge Deployment

---

## 🎨 Visual Overview

<div align="center">
<img src="./svg_figs/model_optimization.svg" alt="Model Optimization" width="100%"/>
</div>

---

## 📊 Quantization

```python
import torch.quantization as quant

# Post-training static quantization
model.eval()
model.qconfig = quant.get_default_qconfig('fbgemm')
quant.prepare(model, inplace=True)

# Calibrate with representative data
for data in calibration_loader:
    model(data)

quant.convert(model, inplace=True)

# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.LSTM}, dtype=torch.qint8
)
```

### Quantization-Aware Training

```python
model.train()
model.qconfig = quant.get_default_qat_qconfig('fbgemm')
quant.prepare_qat(model, inplace=True)

# Train with fake quantization
for epoch in range(epochs):
    for data, target in train_loader:
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

quant.convert(model.eval(), inplace=True)
```

---

## ✂️ Pruning

```python
import torch.nn.utils.prune as prune

# Unstructured pruning (individual weights)
prune.l1_unstructured(module, name='weight', amount=0.3)

# Structured pruning (entire filters)
prune.ln_structured(module, name='weight', amount=0.3, n=2, dim=0)

# Global pruning
parameters_to_prune = [(m, 'weight') for m in model.modules() if hasattr(m, 'weight')]
prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2)

# Remove pruning reparametrization
prune.remove(module, 'weight')
```

---

## 📚 Knowledge Distillation

```python
class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        # Hard label loss
        hard_loss = self.ce(student_logits, labels)
        
        # Soft label loss (distillation)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl(soft_student, soft_teacher) * (self.temperature ** 2)
        
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss
```

---

## 📦 ONNX Export

```python
import torch.onnx

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=11
)

# Run with ONNX Runtime
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
output = session.run(None, {'input': input_data})
```

---

## ⚡ TensorRT

```python
import tensorrt as trt

# Build engine from ONNX
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open("model.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB
engine = builder.build_engine(network, config)
```

---

## ❓ Interview Questions & Answers

### Q1: INT8 vs FP16 vs FP32?
| Type | Bits | Range | Speed | Accuracy |
|------|------|-------|-------|----------|
| FP32 | 32 | Full | 1x | Best |
| FP16 | 16 | Reduced | 2-4x | Good |
| INT8 | 8 | ±127 | 4-8x | Needs calibration |

### Q2: Structured vs Unstructured pruning?
| Structured | Unstructured |
|------------|--------------|
| Remove filters/channels | Remove individual weights |
| Hardware friendly | Sparse matrices |
| Actual speedup | Needs sparse support |
| Coarser | Fine-grained |

### Q3: Why knowledge distillation works?
**Answer:**
- Soft labels carry more information than hard labels
- "Dark knowledge": teacher's uncertainty
- Relative probabilities between classes
- Smoother gradients for training

### Q4: ONNX vs TensorRT?
| ONNX | TensorRT |
|------|----------|
| Format/interop | Runtime/optimizer |
| Cross-platform | NVIDIA only |
| No optimization | Heavy optimization |
| Portable | Maximum performance |

### Q5: How to measure inference speed?
**Answer:**
```python
# Warmup
for _ in range(10):
    model(dummy_input)

# Time
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    model(dummy_input)
torch.cuda.synchronize()
latency = (time.time() - start) / 100 * 1000  # ms
```

---

## 📓 Colab Notebooks

| Topic | Link |
|-------|------|
| Quantization | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/tutorials/blob/main/advanced_source/dynamic_quantization_tutorial.ipynb) |
| Pruning | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/tutorials/blob/main/intermediate_source/pruning_tutorial.ipynb) |
| ONNX | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/onnx/tutorials/blob/master/tutorials/PytorchOnnxExport.ipynb) |

---

<div align="center">

**[← Computational Photography](../17_Computational_Photography/) | [🏠 Home](../README.md) | [Ethics & Safety →](../19_Ethics_Safety/)**

</div>
