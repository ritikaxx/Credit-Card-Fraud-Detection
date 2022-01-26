# Credit-Card-Fraud-Detection
Anomaly Detection using Deep AutoEncoders



1.Due to the bottleneck architecture of the neural network, it is forced to learn a condensed representation from which to reproduce the original input.

2.We feed it only normal transactions, which it will learn to reproduce with high fidelity.

3.As a consequence, if a fraud transaction is sufficiently distinct from normal transactions, the auto-encoder will have trouble reproducing it with its learned weights, and the subsequent reconstruction loss will be high.

4.Anything above a specific loss (treshold) will be flagged as anomalous and thus labeled as fraud.
