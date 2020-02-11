### The WorkPlace Barometer

Matt Valley
Insight Data Science Demo project, cohort 2020A
January, 2020
Seattle WA

###

## Getting Started

This is not a software package, and has no setup.py
The python scripts within should be considered works-in-progress.

### Prerequisites
Major dependencies are:
Spacy
scikit-learn
pandas
skmultilearn (optional)
dash
flask


### Overview
The Workplace Barometer is a tool that sorts text by into pre-specified topics.
It was written for a consulting client who wished to summarize the topics present in long-form survey answers,
and to visualize the results in a web-app.  The front-end visualization is a simple bar-chart that
represents the frequency of occurrence for each topic in the text, and a drop-down menu lets the user
select a topic and returns a randomly-selected example survey response from that category.

Under-the-hood, this is a multi-label classifier, using 16 pre-specified classes.
Training data is typically sparse (across categories), and small, therefore this tool bootstraps training data
using regular-expression matching again using a small library of hand-selected tokens for each category.

Text can be embedding using either tf/idf, or contextual embeddings using a pre-trained BERT network.

Classifier training occurs using a semi-supervised algorithm called "tri-training", which is
a method to label new data using the consensus of three different classifiers.

