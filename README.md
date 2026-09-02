# MultiProp-GNN
## A Full-Stack Graph Neural Network Platform for Ionic Liquid Multi-Property Prediction


<p align="center">

A deep learning based full-stack platform for rapid prediction of ionic liquid thermophysical properties.

</p>


---

# Overview

Ionic liquids (ILs) have attracted extensive attention in chemical engineering, energy storage, gas separation, and green chemistry due to their highly tunable physicochemical properties.

However, the enormous structural space of ionic liquids makes traditional experimental screening expensive and time-consuming.

To address this challenge, this project develops **MultiProp-GNN**, a graph neural network based prediction framework combined with a full-stack interactive web platform for efficient ionic liquid property estimation.

The system integrates:

- Molecular graph neural networks
- RDKit molecular feature extraction
- PyTorch Geometric deep learning framework
- Flask inference backend
- React interactive frontend


Users can directly input ionic liquid molecular structures using SMILES representations and obtain predicted thermophysical properties under specified temperature and pressure conditions.


---

# Supported Properties

MultiProp-GNN focuses on **multi-property prediction**, where different trained models are deployed for different ionic liquid properties.


Currently supported:


| Property | Symbol | Unit |
| :--- | :---: | :--- |
| Mass Density | ρ | kg/m³ |
| Heat Capacity | Cp | J/(mol·K) |
| Surface Tension | σ | mN/m |
| CO₂ Solubility | x | mol/mol |


---

# System Architecture


The platform adopts a front-end and back-end separated architecture:


```
                 User
                  |
                  |
            React Frontend
                  |
                  |
             REST API
                  |
                  |
            Flask Backend
                  |
                  |
          MultiProp-GNN Model
                  |
                  |
        RDKit + PyTorch Geometric

```


The workflow is:


```
SMILES Input

      |

Molecular Graph Construction

      |

Feature Extraction

      |

Graph Neural Network Encoding

      |

Feature Fusion

      |

Property Prediction

      |

Visualization / Result Display

```


---

# Core Features


## 1. Molecular Graph Representation


The platform automatically converts ionic liquid molecular structures into graph representations.


Node features include:


- Atomic number
- Formal charge
- Hydrogen information
- Ring information
- Atomic mass
- Electronegativity
- Hybridization state


Molecular-level features include:


- Morgan fingerprints
- Molecular weight
- TPSA descriptors


Implemented using:


- RDKit
- RDKit Fingerprint Generator


---

## 2. Graph Neural Network Prediction


The prediction models are based on Graph Attention Networks (GAT).


The architecture contains:


### Molecular Encoder


```
Molecular Graph

      |

   GATConv

      |

Graph Embedding

```


### Feature Fusion


The learned graph representation is combined with:


- Molecular fingerprints
- Thermodynamic conditions
- Molecular descriptors


```
Graph Features

+

Fingerprint Features

+

Temperature / Pressure

+

Physical Descriptors

```


### Regression Module


```
Fully Connected Network

          |

Property Prediction

```


---

# Project Structure


```
MultiProp-GNN-for-ILs

│
├── backend
│
│   ├── app.py
│   │      Flask API service
│   │
│   ├── models_lib.py
│   │      GNN models and molecular feature extraction
│   │
│   └── __init__.py
│
│
├── frontend
│
│   ├── src
│   │
│   │── components
│   │      Header.jsx
│   │      PropertyCard.jsx
│   │      ResultPanel.jsx
│   │
│   │── pages
│   │      Home.jsx
│   │      Density.jsx
│   │      Heat.jsx
│   │      Surface.jsx
│   │      Solubility.jsx
│   │
│   │── styles
│   │      global.css
│   │      home.css
│   │      prediction.css
│   │
│   ├── package.json
│   └── vite.config.js
│
│
├── weights
│      Trained model weights
│
│
├── architecture
│
├── requirements.txt
│
└── README.md

```


---

# Environment Requirements


## Backend


Recommended environment:


```
Python >= 3.10

PyTorch >= 2.0

CUDA (optional)

```


Main dependencies:


```
torch

torch-geometric

rdkit

flask

flask-cors

numpy

scikit-learn

joblib

```


---

## Frontend


Required:


```
Node.js >= 18

npm >= 9

```


Frontend framework:


```
React

Vite

React Router

```


---

# Installation


## 1. Clone Repository


```bash
git clone https://github.com/Skywalkeryeahyeah/MultiProp-GNN-for-ILs.git

cd MultiProp-GNN-for-ILs
```


---

# Backend Deployment


## 1. Create Conda Environment


```bash
conda create -n il_gnn python=3.10

conda activate il_gnn
```


---

## 2. Install Python Dependencies


```bash
pip install -r requirements.txt
```


For PyTorch Geometric installation, please follow:


https://pytorch-geometric.readthedocs.io/


---

## 3. Prepare Model Weights


Place trained model files into:


```
weights/


├── density

│   ├── best_cp_model.pth

│   ├── scaler files


├── heat

│   ├── best_cp_model.pth


├── surface

│   ├── best_model.pth


└── solubility

    ├── best_model_co2.pth

```


---

## 4. Start Backend Service


Run:


```bash
python backend/app.py
```


Successful startup:


```
>>> IL-GNN prediction service started

Running on http://127.0.0.1:5000

```


---

# Frontend Deployment


Open another terminal:


```bash
cd frontend
```


Install dependencies:


```bash
npm install
```


Start development server:


```bash
npm run dev
```


The frontend will be available at:


```
http://localhost:5173

```


---

# Usage


## 1. Open Web Interface


```
http://localhost:5173

```


The homepage provides different property prediction modules:


- Mass Density
- Heat Capacity
- Surface Tension
- Gas Solubility


---

## 2. Input Molecular Information


Users provide:


```
Cation SMILES

Anion SMILES

Temperature

Pressure

```


---

## 3. Run Prediction


Click:


```
RUN GNN INFERENCE

```


The system automatically performs:


1. SMILES parsing

2. Molecular graph construction

3. Feature extraction

4. GNN inference

5. Result return


---

# API Documentation


## Prediction Endpoint


```
POST /predict

```


Request example:


```json
{
    "type": "density",

    "cation_smiles": "CCCCn1cc[n+](C)c1",

    "anion_smiles": "F[P-](F)(F)(F)(F)F",

    "T": 298.15,

    "P": 0.1,

    "X": 1.0
}

```


Response:


```json
{
    "status": "success",

    "value": 1.2345
}

```


---

# Model Details


## Molecular Encoder


Graph Attention Network:


```
Input Graph

     |

 GAT Layers

     |

Graph Representation

```


---

## Feature Fusion


The final prediction input combines:


```
Graph embedding

+

Morgan fingerprint

+

Thermodynamic conditions

+

Molecular descriptors

```


---

## Prediction Head


```
Fusion Layer

      |

MLP Regression

      |

Property Value

```


---

# Future Development


Planned improvements:


- Dynamic temperature-property curves

- Dynamic pressure-property curves

- Prediction history management

- Model uncertainty estimation

- Online deployment

- Additional ionic liquid properties


---

# Citation


If this project is useful for your research, please cite:


```
MultiProp-GNN: A Versatile GNN Framework and Full-Stack Platform for
High-Precision Ionic Liquid Property Prediction

```


---

# License


This project is released for academic research purposes.
