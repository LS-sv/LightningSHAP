<div align="center">
<p align="center">
  <img heith=250 width=250 src="images\lshap.png">
</p>

# LightningSHAP: A Cost-Effective Approach to Shapley Values Estimation

### Under Review 
</div>

## Abstract

LightningSHAP is a novel neural network-based estimator for Shapley Values that significantly reduces computational costs while maintaining high explanation quality. Unlike previous approaches that require separate networks for conditional expectations and SV estimation, LightningSHAP uses a unified architecture that jointly computes both, resulting in 25%-60% faster training+inference times on tabular datasets and 30%-55% speedup on image datasets, with comparable or lower estimation errors than existing methods. This approach is particularly valuable for scenarios requiring frequent model retraining, where traditional SV estimators like FastSHAP become prohibitively expensive. By minimizing feature-level estimation errors while preserving SV efficiency, LightningSHAP makes explainable AI more practical for real-world applications.





