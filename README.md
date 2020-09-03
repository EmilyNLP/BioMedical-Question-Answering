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
