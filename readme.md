
# Beat Tracking and Tempo Estimation

## Proposal 

Beat tracking tries to detect the position of beats in an audio signal while tempo estimation is about determining the overall period of these beats in BPM. These are fundamental to a lot of applications including music transcription, DJ systems, synchronization and interactive performance. 

Despite tons of research, tempo estimation remains a challenge due to the expressive timing, tempo drift, genre variation, weak percussion and things like octave errors where systems predict fractions of the correct tempo. A lot of modern approaches use signal processing and deep learning to improve accuracy. 

In this project we aim to design and evaluate a supervised learning approach for tempo estimation using an annotated dataset. We will compare the neural models with classic machine learning baselines and analyze their behaviour understanding evaluation metrics. Emphasis will be put on reproduction, fair comparison under our EDM genre.

## Objectives

The primary objective of this project is to compare deep learning and classical machine learning approaches for tempo estimation. Beyond achieving high accuracy and robustness, we aim to understand how design choices affect model performance and which features contribute most to accurate tempo predictions. 

## Timeline

Week 1: Data ingestion, annotation parsing and exploring statistics.
Week 2: Feature extraction pipeline and baseline regression model.
Week 3: Neural implementation and initial training.
Week 4: Hyperparameter optimization and studies.
Week 5: Final evaluation, comparison with reference systems
Week 6: Paper writing, figures and presentation preparation.

## Tools and Datasets

This project leverages the GiantSteps Tempo Dataset, which contains 664 annotated 2-minute audio previews. Audio data is loaded, resampled, and analyzed with librosa, including computation of the Zero-Crossing Rate for feature extraction. Data preprocessing and manipulation are performed using NumPy and pandas. The CNN model is implemented in PyTorch, while madmom and BeatNet provide pretrained baseline tempo estimates. For classical approaches, a Random Forest Regressor is implemented using scikit-learn. The dataset is split into 80% for training and 20% for testing to evaluate model performance. Standard metrics such as accuracy and MAE will be used for evaluation. 

## Team Member Roles

### Dylan

Dylan will be responsible for implementing the CNN, and evaluating and comparing the two models. 

### Asa

Asa will be responsible for preparing the dataset for the pipeline, creating visualizations, and preparing the presentation structure.

### Connor

Connor will be responsible for implementing the feature extraction method, implementing the RF regressor, and writing the ISMIR paper.

Connor
Objective: Write ISMIR Paper
PI1: Abstract (Expected)
PI2: Methods (Expected)
PI3: Visualizations (Expected)
PI4: Results (Expected)
PI5: Discussion (Expected)


Objective: CNN Comparison with RF Regressor
PI1 (Basic): Feature Extraction (Zero Crossing Rate via Librosa) (Advanced) 
PI2 (Expected): Define and train RF Regressor model using Scikit-learn 
PI3 (Advanced): Tune hyperparameters via Optuna
PI4 (Advanced): K-fold cross-validation 
PI5 (Expected): Implement same evaluation metrics as NN 

Asa
Objective: Prepare GiantSteps dataset
PI1: Load GiantSteps audio files (Basic)
PI2: Load GiantSteps tempo annotations (Basic) 
PI3: Combine audio + tempo annotations into dict (data + label) (Expected)
PI4: Normalize audio (sampling rate, trim length to 120s) (Expected)
PI5: Identify and handle missing data (Expected)

Objective: Visualization and Interpretability
PI1 (Basic): Plot predicted vs. ground truth BPM for test tracks
PI2 (Basic): Plot error histograms and octave error distributions
PI3 (Expected): Visualize model learning curves and validation metrics over epochs
PI4 (Expected): Visualize feature importance (e.g., via RF or activation maps in CNN)
PI5 (Advanced): Include all visualizations in ISMIR-ready figures with clear captions and analysis

Dylan
Objective: Track training and evaluation results systematically
PI1 (Basic): Log model loss and accuracy per epoch
PI2 (Basic): Save hyperparameters for each run
PI3 (Expected): Save model checkpoints
PI4 (Expected): Compare multiple experiments easily
PI5 (Advanced): Generate summary tables and charts for ISMIR figures

Objective: Develop and Evaluate a Supervised BPM Estimation Model
PI1 (Basic): Implement baseline neural network architecture in PyTorch (regression)
PI2 (Basic): Train model on GiantSteps training split and verify convergence
PI3 (Expected): Implement standard tempo evaluation metrics (accuracy, MAE)
PI4 (Expected): Perform hyperparameter tuning via Optuna and report impact on validation performance
PI5 (Expected): Plot training and val loss over epochs


## Related Work:
Gouyon, Fabien, et al. "An experimental comparison of audio tempo induction algorithms." IEEE Transactions on Audio, Speech, and Language Processing 14.5 (2006): 1832-1844.

Percival, Graham, and George Tzanetakis. "Streamlined tempo estimation based on autocorrelation and cross-correlation with pulses." IEEE/ACM Transactions on Audio, Speech, and Language Processing 22.12 (2014): 1765-1776.

Böck, Sebastian, and Markus Schedl. "Enhanced beat tracking with context-aware neural networks." Proc. Int. Conf. Digital Audio Effects. 2011. 

F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.

A. S. Chowdhury, M. I. Hossain, and Y. Wang, “BeatNet: CRNN and Transformer for Joint Beat and Downbeat Estimation,” arXiv preprint arXiv:2108.03576, 2021.

H. Schreiber and M. Müller, “A Single-Step Approach to Musical Tempo Estimation Using a Convolutional Neural Network,” in Proceedings of the International Society for Music Information Retrieval Conference (ISMIR), 2015.

F. Krebs, S. Böck, and G. Widmer, “Rhythmic Pattern Modeling for Beat and Downbeat Tracking in Musical Audio,” arXiv preprint arXiv:2011.02619, 2020.

M. J. Hydri and G. Tzanetakis, “1D State Space Models for Efficient Music Rhythm Analysis,” arXiv preprint arXiv:2111.00704, 2021.

S. Böck, F. Krebs, and G. Widmer, “madmom: A New Python Audio and Music Signal Processing Library,” in Proceedings of the ACM International Conference on Multimedia, 2016.


## Bibliography

F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.

S. van der Walt, S. C. Colbert, and G. Varoquaux, “The NumPy Array: A Structure for Efficient Numerical Computation,” *Computing in Science & Engineering*, vol. 13, no. 2, pp. 22–30, 2011.

A. Paszke et al., “PyTorch: An Imperative Style, High-Performance Deep Learning Library,” in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, pp. 8024–8035, 2019.

S. Böck, F. Krebs, and G. Widmer, “madmom: A New Python Audio and Music Signal Processing Library,” in *Proceedings of the ACM International Conference on Multimedia*, 2016.

B. McFee et al., “librosa: Audio and Music Signal Analysis in Python,” in *Proceedings of the Python in Science Conference (SciPy)*, 2015.

A. S. Chowdhury, M. I. Hossain, and Y. Wang, “BeatNet: CRNN and Transformer for Joint Beat and Downbeat Estimation,” arXiv preprint arXiv:2108.03576, 2021.

A. S. Chowdhury, M. I. Hossain, and Y. Wang, “BeatNet: CRNN and Transformer for Joint Beat and Downbeat Estimation,” arXiv preprint arXiv:2108.03576, 2021.

H. Schreiber and M. Müller, “A Single-Step Approach to Musical Tempo Estimation Using a Convolutional Neural Network,” in Proceedings of the International Society for Music Information Retrieval Conference (ISMIR), 2015.

F. Krebs, S. Böck, and G. Widmer, “Rhythmic Pattern Modeling for Beat and Downbeat Tracking in Musical Audio,” arXiv preprint arXiv:2011.02619, 2020.

M. J. Hydri and G. Tzanetakis, “1D State Space Models for Efficient Music Rhythm Analysis,” arXiv preprint arXiv:2111.00704, 2021.

Gouyon, Fabien, et al. "An experimental comparison of audio tempo induction algorithms." IEEE Transactions on Audio, Speech, and Language Processing 14.5 (2006): 1832-1844.

Percival, Graham, and George Tzanetakis. "Streamlined tempo estimation based on autocorrelation and cross-correlation with pulses." IEEE/ACM Transactions on Audio, Speech, and Language Processing 22.12 (2014): 1765-1776.

Böck, Sebastian, and Markus Schedl. "Enhanced beat tracking with context-aware neural networks." Proc. Int. Conf. Digital Audio Effects. 2011. 

J. P. Bello, L. Daudet, S. Abdallah, C. Duxbury, M. Davies, and M. B. Sandler, “A tutorial on onset detection in music signals,” IEEE Transactions on Speech and Audio Processing, vol. 13, no. 5, pp. 1035–1047, 2005.

A. Klapuri, “Multiple fundamental frequency estimation based on harmonicity and spectral smoothness,” IEEE Transactions on Speech and Audio Processing, vol. 11, no. 6, pp. 804–816, 2003.

D. P. W. Ellis, “Beat tracking by dynamic programming,” Journal of New Music Research, vol. 36, no. 1, pp. 51–60, 2007.

M. Müller, T. Vogl, and F. Kurth, “An efficient approach to tempo induction,” in Proceedings of the International Society for Music Information Retrieval (ISMIR), 2006.

J. P. Bello, L. Daudet, and M. Davies, “Rhythm patterns in musical audio,” in Proceedings of the International Society for Music Information Retrieval (ISMIR), 2004.

G. Peeters, A. Gouyon, J. L. Rodet, and S. Dixon, “Relative tempo estimation based on periodicity detection and beat tracking,” in Proceedings of the International Symposium on Music Information Retrieval (ISMIR), 2002.

T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, “Optuna: A Next-generation Hyperparameter Optimization Framework,” arXiv preprint arXiv:1907.10902, 2019.

B. McFee, C. Raffel, D. Liang, D. P. W. Ellis, M. McVicar, E. Battenberg, and O. Nieto, “librosa: Audio and Music Signal Analysis in Python,” in *Proceedings of the 14th Python in Science Conference (SciPy)*, Austin, TX, USA, 2015, pp. 18–24, doi:10.25080/Majora-7b98e3ed-003.












