# Predicting Recurrent Chat Contact in a Low-Threshold Psychological Intervention for Children and Young Adults: A Natural Language Processing Approach
### Silvan Hornstein, Jonas Scharfenberger, Ulrike Lueken, Richard Wundrack, Kevin Hilbert
### Abstract
Chat-based counseling hotlines emerged as a promising low-threshold intervention for youth mental health. However, despite the resulting availability of large text corpora, little work has investigated Natural Language Processing (NLP) applications within this setting. Therefore, this preregistered approach (OSF: XA4PN) utilizes a sample of approximately 19,000 children and young adults that received a chat consultation from a 24/7 crisis service in Germany. Around 800,000 messages were used to predict whether chatters would contact the service again, as this would allow the provision of or redirection to additional treatment. We trained an XGBoost Classifier on the words of the anonymized conversations, using repeated cross validation and bayesian optimization for hyperparameter search. The best model was able to achieve an AUROC score of 0.68 (p < .01) on the previously unseen 3,962 newest consultations. A shapely-based explainability approach revealed that words indicating younger age or female gender and terms related to self harm and suicidal thoughts were associated with higher chance of recontacting. We conclude that NLP-based predictions of recurrent contact are a promising path towards personalized care at chat hotlines.

### This Repository contains
 2. Algorithm Training Code, including Code for Figure 1 (ROC Graph): training.ipynb <br/>
 3. Final Prediction Code, including Code for Figure 2 (Confusion Matrix Graph): evaluation.ipynb <br/>
 4. Code for the SHAP-based interpretability approach, including Code For Figure 3 (SHAP Plot.): shap_values.ipybn <br/>
 5. requirements.txt, containing the used package versions.
 6. utils.py with the used permutation function (from https://github.com/qbarthelemy/PyPermut/blob/main/examples/compute_auroc_pvalue.py)

### Links:

OSF Registration: https://osf.io/xa4pn <br/>
Preprint: https://www.researchsquare.com/article/rs-3407849/v1



