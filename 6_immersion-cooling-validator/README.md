# 🌡️ Immersion Cooling Validator (ICV)

[![CI/CD](https://github.com/bosco-chiramel/immersion-cooling-validator/actions/workflows/ci.yml/badge.svg)](https://github.com/bosco-chiramel/immersion-cooling-validator/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Monte Carlo validation framework for immersion cooling fluids against OCP specifications for AI data centers.

## ✨ Features

- **Monte Carlo Engine**: N=10,000 samples with reproducible results
- **OCP Compliance**: Thermal, electrical, lifetime validation
- **Economic Analysis**: 83-98% cost savings vs imported fluids
- **Sensitivity Analysis**: Sobol indices for parameter importance
- **Web Interface**: React dashboard with real-time charts
- **REST API**: FastAPI backend

## 🚀 Quick Start

```bash
pip install -e .

# Python usage
from icv import validate_fluid
result = validate_fluid(n_samples=10000)
print(result.summary())
```

## 🖥️ Web App

```bash
# Docker
docker build -t icv-web . && docker run -p 8000:8000 icv-web

# Or directly
cd web/backend && uvicorn main:app --port 8000
```

## 📊 OCP Requirements

| Parameter | Requirement |
|-----------|-------------|
| Junction Temp | < 88°C |
| Breakdown Voltage | > 45 kV |
| Volume Resistivity | > 10¹¹ Ω·cm |
| P5 Service Life | > 5 years |

## 📈 API

- `POST /api/simulate/quick` - Quick simulation
- `GET /api/sensitivity` - Sensitivity analysis
- `POST /api/economics/forex` - Forex projection
- Full docs at `/docs`

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📚 Citation

Chiramel, B. (2024). "Monte Carlo Validation Framework for Group III Hydrocarbon-Based Single-Phase Immersion Cooling Fluid in AI Data Center Applications"

## 📄 License

MIT
