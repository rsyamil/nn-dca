# nn-dca

Static and dynamic versions of Physics-guided deep learning (PGDL) with embedded statistical or empirical physics model and residual learning model to compensate for the imperfect physics model.  

```
nn-dca
├── static
│   └── pgdl-empirical-residual-geomech.ipynb
└── dynamic
    └── dpgdl-empirical-residual-geomech.ipynb
```

## Workflow

In physics-constrained neural network models (i.e., predictive models with embedded physical functions shown as F1F2), the network predictive performance can be degraded when the embedded physics (F2) does not represent the relationship within the observed data. A residual learning model (F3) is introduced to improve the predictions from a physics-constrained neural network, whereby an auxiliary neural network component is introduced to compensate for the imperfect description of the constraining physics. 

![Workflow](/readme/workflow.png)

When a dataset cannot be fully represented by a trained physics-constrained neural network model due to the lack of representativeness of the embedded physics F2, the predictions come with a large error or residual when compared to the ground truth. F3 is employed to learn the complex spatial and temporal correspondence between the well properties such as formation and completion parameters to the expected residuals. The proposed method results in a final prediction that combines the prediction from the physics-constrained neural network F1F2 with the predicted residual from F3.
