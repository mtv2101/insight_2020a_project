### The WorkPlace Barometer

Matt Valley
<br>Insight Data Science Demo project, cohort 2020A
<br>Seattle WA

###

## Getting Started

This is not a software package, and has no setup.py
The python scripts within should be considered works-in-progress.

### Prerequisites
Major dependencies are:
Spacy <br>
scikit-learn<br>
pandas<br>
skmultilearn (optional)<br>
dash<br>
flask


### Overview
<p>The Workplace Barometer is a tool that sorts text by into one or many pre-specified topics.
It was written for a consulting client who wished to summarize the topics present in long-form survey answers,
and to visualize the results in a web-app.  The front-end visualization is a simple bar-chart that
represents the frequency of occurrence for each topic in the text, and a drop-down menu lets the user
select a topic and returns a randomly-selected example survey response from that category.</p>

<p>Under-the-hood, this is a multi-label classifier, using 16 pre-specified classes.
Training data is typically sparse (across categories), and small, therefore this tool bootstraps training data
using regular-expression matching again using a small library of hand-selected tokens for each category.</p>

<p>Text can be embedding using either tf/idf, or contextual embeddings using the pre-trained BERT network.</p>

<p>Classifier training occurs using a semi-supervised algorithm called "tri-training", which is
a method to label new data using the consensus of three different classifiers.</p>

<p> refs:<br>
[original paper:](https://ieeexplore.ieee.org/document/1512038/)<br>
[recent SOA implementation](https://arxiv.org/abs/1804.09530)</p>


