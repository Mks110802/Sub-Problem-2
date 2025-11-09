# Sub-Challenge 2: RAG-Powered Nodal Analysis
## Complete Setup & Implementation Guide

---

## 📋 Overview

This implementation creates an **automated well production capacity estimation system** using:
- **RAG (Retrieval Augmented Generation)** for parameter extraction
- **Nodal Analysis** for production calculations
- **Agentic workflow** orchestrating the entire process

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG-Nodal Analysis Agent                  │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌──────────────┐   ┌──────────────┐
│  Document     │   │  Parameter   │   │   Nodal      │
│  Processor    │──▶│  Extractor   │──▶│  Analysis    │
└───────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
  [ChromaDB]          [Ollama]          [NumPy/SciPy]
```

---

## 📦 Installation

### 1. Create Project Structure

```bash
cd geohackathon-2025

# Create necessary directories
mkdir -p src examples outputs tests chroma_db

# Copy the implementation files
# - nodal_analysis_rag.py → src/
# - rag_nodal_integration.py → examples/sub_challenge_3.py
```

### 2. Install Dependencies

```bash
# Update requirements.txt
cat >> requirements.txt << 'EOF'

# Sub-Challenge 3: Nodal Analysis
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0

# Already in requirements (verify):
# docling[rapidocr]
# sentence-transformers
# chromadb
# ollama
EOF

# Install
pip install -r requirements.txt
```

### 3. Install and Setup Ollama

```bash
# Download from https://ollama.ai
# Then pull the model:
ollama pull llama3.2:3b

# Verify installation
ollama list
```

---

## 🚀 Usage

### Quick Start

```bash
# Run complete RAG-Nodal analysis workflow
python examples/sub_challenge_3.py --well "NLW-GT-03"
```

### Step-by-Step Execution

```python
from pathlib import Path
from examples.sub_challenge_3 import RAGNodalAnalysisAgent

# Initialize agent
agent = RAGNodalAnalysisAgent(
    well_name="NLW-GT-03",
    data_folder=Path("Training data-shared with participants")
)

# Run complete workflow
report = agent.run()

# Access results
print(f"Oil Rate: {report['results']['operating_point']['oil_rate']} bbl/day")
print(f"PI: {report['results']['metrics']['productivity_index']}")
```

---

## 🧪 Testing

### Create Test Suite

Create `tests/test_nodal_analysis.py`:

```python
import pytest
import numpy as np
from src.nodal_analysis_rag import WellParameters, NodalAnalysis

@pytest.fixture
def sample_params():
    return WellParameters(
        measured_depth=3500.0,
        true_vertical_depth=3450.0,
        inner_diameter=5.5,
        reservoir_pressure=4500.0,
        reservoir_temperature=350.0,
        permeability=50.0,
        skin_factor=2.5,
        oil_api_gravity=35.0,
        gas_specific_gravity=0.65,
        water_cut=0.2,
        gor=500.0,
        wellhead_pressure=200.0,
        separator_pressure=100.0
    )

def test_well_parameters_validation(sample_params):
    """Test parameter validation"""
    assert sample_params.measured_depth >= sample_params.true_vertical_depth
    assert 0 <= sample_params.water_cut <= 1
    
def test_ipr_calculation(sample_params):
    """Test IPR curve calculation"""
    nodal = NodalAnalysis(sample_params)
    pressures = np.linspace(100, 4000, 10)
    rates = nodal.calculate_ipr(pressures)
    
    # Rates should decrease with pressure
    assert all(rates[i] >= rates[i+1] for i in range(len(rates)-1))
    
def test_production_capacity_estimation(sample_params):
    """Test complete production capacity estimation"""
    nodal = NodalAnalysis(sample_params)
    results = nodal.estimate_production_capacity()
    
    # Check result structure
    assert "operating_point" in results
    assert "metrics" in results
    assert "units" in results
    
    # Check reasonable values
    op = results["operating_point"]
    assert op["total_liquid_rate"] > 0
    assert op["oil_rate"] > 0
    assert op["bottom_hole_pressure"] > 0

def test_productivity_index_calculation(sample_params):
    """Test PI calculation"""
    nodal = NodalAnalysis(sample_params)
    pi = nodal._calculate_productivity_index()
    
    # PI should be positive
    assert pi > 0
    
def test_rag_extractor():
    """Test RAG parameter extraction"""
    from examples.sub_challenge_3 import DocumentProcessor
    
    processor = DocumentProcessor()
    
    # Test query
    results = processor.query("What is the measured depth?", n_results=3)
    
    assert len(results) <= 3
    assert all("text" in r for r in results)
```

### Run Tests

```bash
# Run all tests
pytest tests/test_nodal_analysis.py -v

# Run with coverage
pytest tests/test_nodal_analysis.py --cov=src --cov-report=html

# Run specific test
pytest tests/test_nodal_analysis.py::test_production_capacity_estimation -v
```

---

## 📊 Expected Output

```
======================================================================
RAG-Nodal Analysis Agent: NLW-GT-03
======================================================================

Step 1: Indexing well documents...
Processing: Well_Completion_Report.pdf
Processing: Production_Test_Results.pdf
✓ Indexed 1247 chunks

Step 2: Extracting parameters via RAG...
  Extracting geometry parameters...
  Extracting reservoir parameters...
  Extracting fluid parameters...
  Extracting production parameters...
  ✓ Parameter extraction complete

Step 3: Performing nodal analysis...
  ✓ Nodal analysis complete

Step 4: Generating report...
  ✓ Report saved to: outputs/nodal_analysis_NLW-GT-03.json

======================================================================
ANALYSIS SUMMARY
======================================================================

Well: NLW-GT-03

Production Capacity:
  Total Liquid: 1247.56 bbl/day
  Oil Rate:     998.05 bbl/day
  Gas Rate:     499025.0 scf/day

Key Metrics:
  Productivity Index: 0.89 bbl/day/psi
  Water Cut:          20.0%

Recommendations:
  1. Well is operating within normal parameters.
  2. Consider optimization of wellhead pressure for increased production.

======================================================================
```

---

## 📁 Output Files

### 1. JSON Report (`outputs/nodal_analysis_NLW-GT-03.json`)

```json
{
  "well_name": "NLW-GT-03",
  "analysis_date": "2025-01-04",
  "parameters": {
    "measured_depth": 3500.0,
    "true_vertical_depth": 3450.0,
    "inner_diameter": 5.5,
    ...
  },
  "results": {
    "operating_point": {
      "total_liquid_rate": 1247.56,
      "oil_rate": 998.05,
      "water_rate": 249.51,
      "gas_rate": 499025.0,
      "bottom_hole_pressure": 2847.23
    },
    "metrics": {
      "productivity_index": 0.89,
      "drawdown": 1652.77,
      "water_cut": 20.0
    }
  },
  "recommendations": [...]
}
```

---

## 🎯 Key Features

### ✅ Implemented

1. **Document Processing**
   - PDF parsing with OCR (Docling + RapidOCR)
   - Text chunking with overlap
   - Vector embedding (nomic-embed-text-v1.5)

2. **RAG System**
   - ChromaDB for vector storage
   - Semantic search for parameter retrieval
   - LLM-based extraction (Ollama + Llama 3.2)

3. **Parameter Extraction**
   - Well geometry (MD, TVD, ID)
   - Reservoir properties (pressure, temp, permeability, skin)
   - Fluid properties (API gravity, GOR, water cut)
   - Production constraints (wellhead & separator pressure)

4. **Nodal Analysis**
   - IPR curve calculation (Vogel's equation)
   - TPR curve calculation
   - Operating point determination
   - Production capacity estimation

5. **Reporting**
   - Structured JSON output
   - Performance metrics
   - Automated recommendations

---

## 🔧 Customization

### Adjust Model Parameters

```python
# In rag_nodal_integration.py

class LLMParameterExtractor:
    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model  # Change to "llama3.2:1b" for faster inference
```

### Modify Nodal Analysis Correlations

```python
# In nodal_analysis_rag.py

class NodalAnalysis:
    def calculate_tpr(self, flow_rates):
        # Replace with Beggs-Brill or Hagedorn-Brown
        # for more accurate pressure drop calculations
        pass
```

### Add More Parameters

```python
# In rag_nodal_integration.py - _extract_all_parameters()

queries["completion"] = {
    "query": "What completion type and sand control method were used?",
    "params": ["completion_type", "sand_control"]
}
```

---

## 🐛 Troubleshooting

### Issue: ChromaDB Connection Error

```bash
# Solution: Clear and reinitialize
rm -rf chroma_db/
python examples/sub_challenge_3.py --well "NLW-GT-03"
```

### Issue: Ollama Model Not Found

```bash
# Solution: Pull model again
ollama pull llama3.2:3b
ollama list  # Verify
```

### Issue: Parameter Extraction Returns Defaults

**Cause**: Documents not properly indexed or LLM response parsing failed

**Solution**:
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 📈 Performance Optimization

### 1. Batch Processing

```python
# Process multiple wells
wells = ["NLW-GT-03", "NLW-GT-04", "NLW-GT-05"]

for well in wells:
    agent = RAGNodalAnalysisAgent(well, data_folder)
    agent.run()
```

### 2. Caching

```python
# Cache embeddings
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text: str):
    return embedding_model.encode([text])[0]
```

### 3. Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_well, w) for w in wells]
    results = [f.result() for f in futures]
```

---

## 🎓 Evaluation Criteria (30% of Grade)

### Scoring Breakdown:

1. **Parameter Extraction Accuracy** (10%)
   - Correct extraction of MD, TVD, ID
   - Correct extraction of reservoir properties
   - Correct extraction of fluid properties

2. **Nodal Analysis Implementation** (10%)
   - Valid IPR/TPR calculations
   - Accurate operating point determination
   - Reasonable production estimates

3. **RAG System Integration** (5%)
   - Proper document indexing
   - Effective semantic search
   - Successful LLM integration

4. **Code Quality** (5%)
   - Type hints and documentation
   - Test coverage
   - Error handling

---

## 📚 References

### Nodal Analysis Theory
- **Vogel's IPR**: Vogel, J.V. (1968). "Inflow Performance Relationships for Solution-Gas Drive Wells"
- **Multiphase Flow**: Beggs & Brill correlation for pressure drop
- **Productivity Index**: Darcy's law for radial flow

### RAG Implementation
- **Embeddings**: [nomic-embed-text-v1.5 docs](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- **Vector DB**: [ChromaDB documentation](https://docs.trychroma.com/)
- **LLM**: [Ollama documentation](https://ollama.ai/docs)

---

## 🤝 Contributing

For team collaboration:

```bash
# Create feature branch
git checkout -b feature/nodal-improvements

# Make changes
git add src/nodal_analysis_rag.py
git commit -m "feat: improve IPR calculation accuracy"

# Push and create PR
git push origin feature/nodal-improvements
```

---

## ✅ Checklist

Before submission, ensure:

- [ ] All dependencies installed
- [ ] Ollama model downloaded
- [ ] Documents indexed in ChromaDB
- [ ] Tests passing (`pytest tests/`)
- [ ] Code formatted (`black src/ examples/`)
- [ ] Type hints complete (`mypy src/`)
- [ ] Documentation updated
- [ ] Example outputs generated
- [ ] Demo video recorded (<10 min)
