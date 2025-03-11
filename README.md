<div align="center">
<p align="center">
  <img heith=350 width=350 src="images\lightningshap.jpg">
</p>
## LightningSHAP: A Cost-Effective Approach to Shapley Values Estimation

### Under Review 
</div>
Shapley Values (SVs) are concepts established for explaining black-box machine learning models by quantifying feature contributions to model predictions. 
Since their computation has exponential complexity in the number of features, a variety of approximated approaches to SV estimation have been proposed. 
The state-of-the-art neural SVs estimator (FastSHAP) advances sampling-based methods by first training a supervised surrogate model to learn the conditional expectation of the original model given every feature subset and then generating SVs estimates through a separate network using weighted least squares. 
While ensuring fast SV inference, this approach requires a significant training time, thus becoming unsuitable for scenarios where the black-box model has to be frequently retrained.
To address this limitation, we propose \textsc{LightningSHAP}, a cost-effective neural network-based estimator that jointly computes conditional expectations and SVs estimations to reduce training cost.
The unified network learning process minimizes feature-level estimation errors while preserving SVs efficiency.
Experiments run on both dynamic and static tabular datasets show that \textsc{LightningSHAP} achieves a 25\%-60\% speedup in training+inference times compared to existing neural SV estimators and lower or comparable estimation errors. Furthermore, the results obtained on image datasets indicate that \textsc{LightningSHAP} yields a 30\%-55\% speedup while preserving explanation quality.
