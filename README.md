<div align="center">
<p align="center">
  <img heith=250 width=250 src="images\ls.png">
</p>

# LightningSHAP: A Cost-Effective Approach to Shapley Values Estimation

### Under Review 
</div>

## Abstract

LightningSHAP is a novel neural network-based estimator for Shapley Values that significantly reduces computational costs while maintaining high explanation quality. Unlike previous approaches that require separate networks for conditional expectations and SV estimation, LightningSHAP uses a unified architecture that jointly computes both, resulting in 25%-60% faster training+inference times on tabular datasets and 30%-55% speedup on image datasets, with comparable or lower estimation errors than existing methods. This approach is particularly valuable for scenarios requiring frequent model retraining, where traditional SV estimators like FastSHAP become prohibitively expensive. By minimizing feature-level estimation errors while preserving SV efficiency, LightningSHAP makes explainable AI more practical for real-world applications.

## Implementation Details

### Tabular Data

**LightningSHAP**
- Architecture: Multi-Layer Perceptron (MLP) with a MaskLayer to perform element-wise multiplication between input features and binary subset masks
- Network structure: Three linear layers with LeakyReLU activation functions
  - First two layers: 512 nodes each
  - Output layer: Dimensionality determined by the product of number of classes and features
- Optimization: AdamW optimizer with learning rate of 2·10⁻⁴ and weight decay of 0.01
- Training: 200 epochs with batch size and oversampling factor of 32, using paired sampling

**FastSHAP (Comparison)**
- Surrogate model: MaskLayer followed by two layers (512 nodes each) and a final layer matching the number of classes, with LeakyReLU activations
- Explainer network: Three layers of 256 nodes each with LeakyReLU activations
- Uses identical hyperparameters as LightningSHAP for fair comparison

**Other Explainers (Comparison)**
- Exact, DeepSHAP, GradientSHAP: SHAP library implementations
- Unbiased KernelSHAP and Permutation: Sourced from ShapReg repository
- Monte Carlo method: 1,000 iterations (empirically determined balance between performance and efficiency)

### Image Data

**LightningSHAP**
- Architecture: MaskLayer for selective pixel masking followed by a pre-trained ResNet50 backbone
- Output: 1x1 convolutional layer generating 14×14 attribution maps for each class
- Optimization: AdamW optimizer (learning rate: 2·10⁻⁴, weight decay: 0.01)
- Training approach: Batch size of 64 with paired sampling and oversampling factor of 4
- Initialization: ImageNet pre-trained weights for ResNet backbone, Kaiming uniform initialization for additional input layer

**FastSHAP (Comparison)**
- Surrogate model: ResNet50 with a MaskLayer for element-wise multiplication between input and subset mask
- Explainer: ResNet50 backbone with final layers replaced by a 1x1 convolutional layer (14×14 superpixel attributions)
- Implementation details: Omits normalization layers to enhance explanation quality, uses ImageNet pre-trained weights

**Other Explainers (Comparison)**
- DeepSHAP and KernelSHAP: SHAP library implementations (KernelSHAP configured for 14×14 pixel attributions per class)
- GradCAM, SmoothGrad, IntegratedGradients: Sourced from TF-Explain repository
- CXPlain: Network parameters identical to FastSHAP for architectural consistency

