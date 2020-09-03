# BioMedical-question-answering
# 1. Introduction

In this project, I aim to leverage general Question Answering(QA) language model for transfer learning of biomedical Question Answering application.I see the Stanford's Question Answering task on dataset([SQuAD](https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/)) that includes more than 150,000 question-answer pairs and their corresponding context paragraphs from Wikipedia as the general QA task. For better match the biomedical QA task, I only use the SQuAD dataset version 1.1. A typical sample of data in SQuAD 1.1 is as following:

question:"What was Maria Curie the first female recipient of?"
<br>
context:"One of the most famous people born in Warsaw was Maria Skłodowska-Curie, who achieved international recognition for her research on radioactivity and was the first female recipient of the Nobel Prize. Famous musicians include Władysław Szpilman and Frédéric Chopin. Though Chopin was born in the village of Żelazowa Wola, about 60 km (37 mi) from Warsaw, he moved to the city with his family when he was seven months old. Casimir Pulaski, a Polish general and hero of the American Revolutionary War, was born here in 1745."
<br>
answer:"Nobel Prize"

I see the  [BioASQ Task 7B](http://participants-area.bioasq.org/datasets/) as the biomedic-domain-specific QA task. There are different types of QA pairs in the BioASQ dtaset: Yes/no questions, Factoid questions, List questions, Summary questions. I will only focus on sloving the task of factoid questions. A sample of factoid question is as below:

question:"Which R package could be used for the identification of pediatric brain tumors?"
<br>
context:"MethPed: an R package for the identification of pediatric brain tumor subtypes"
<br>
answer:"MethPed"

# 2. Approach

My approach for this task is to implement [RoBERTa](https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/) for both general and biomedical QA tasks. The outline of the whole precedures is below.


![image.png](attachment:image.png)


Computing platform: 1 GPU with 15G Memory on Kaggle<br>
Programming language and framework: Python 3, PyTorch 1.6.1, Transfomers 2.11<br>

## 2.1 Pretrain RoBERTa with biomedical publication corpus
In order to have the RoBERTa better represents the biomedical domain context, first, I pretrain the Vanilla RoBERTa model with corpus downloaded from PubMed Abstracts: ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/. This procedure includes web scraping the text of biomedical publications, formatting and tokenizing the text, training RoBERTa with masked language modeling loss.

![image.png](attachment:image.png)



My code for web scraping PubMed corpus and processing text is here.<br>
My code for pretraining RoBERTa is here.<br>

The PubMed Abstracts includes 2.48 billion words. I only download 2.5% of the corpus for pretraining,which is 618M words. The pretraining took more than 3 hours.

## 2.2 Fine-tune RoBERTa QA model with SQuAD and BioASQ dateset 
After I have the pretrained RoBERTa model, I add QA head on the top of it to build RoBERTa QA model. Then, I fine-tune the model with SQuAD dataset, then I fine-tune the model with BioASQ train dataset.The fine-tuning procedure is illustrated as below.

![image.png](attachment:image.png)

The fine-tuning processing is somewhat complicated. Thus I split the task into 2 subtasks.<br>
1) Load the SQuAD data from json file, convert the raw data to dataset which the model are able to take. And I convert the BioASQ data into the same format as the SQuAD dataset. My code is here.<br>
2) Fine-tune the QA model with SQuAD dataset and BioASQ dateset, evaluate the model on the validation datset(including retriving the prediction of answers). The fine-tuning took more than 1 hour. My code is here

3. Experiments results
I use the same evaluation metrics as the SQuAD task, which are Exact Match(EM) and F1 Score. The results of my experiments are listed below.

Model	Pretrained with PubMed corpus(618M words)	Fine-tuned with SQuAD train dateset(89705 samples )	Fine-tuned with BioASQ train dataset(3029 samples)	Evaluated on SQuAD dev dataset(10570 samples)	Evaluated on BioASQ dev dataset(460 samples)
RoBERTa-Base	NO	YES	NO	F1=89.46/EM=82.15	F1=75.57/EM=60
RoBERTa-Base	NO	YES	YES	F1=80.65/EM=71.36	F1=81.68/EM=66.3
RoBERTa-Base	YES	YES	NO	F1=88.95/EM=81.63	F1=76.68/EM=60.65
RoBERTa-Base	YES	YES	YES	F1=78.75.46/EM=69.4	F1=84.88/EM=72.6
Based on the results of my experiments, The model pretrained with PubMed corpus and fine-tuned with SQuQD and BioASQ dataset has the best score while tested on the BioASQ dev dataset. And the model without pretrained with PubMed corpus and only fine-tuned with SQuQD dataset gets the best score while tested on SQuQD dev dataset. 

