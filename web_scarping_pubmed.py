#!/usr/bin/env python
# coding: utf-8


from ftplib import FTP
import os
import gzip
import shutil
import xml.etree.ElementTree as ET 
import pickle


#connect to ftp server
ftp = FTP('ftp.ncbi.nlm.nih.gov')
ftp.login()
ftp.cwd('pubmed')
ftp.cwd('baseline')
filelist=ftp.nlst()
fulllist=[file for file in filelist if ('.gz' in file and '.md5' not in file)]
download_filelist=fulllist[0:len(downloadfile):40] # only download 2.5% files from ftp server 


# The directory to store the file 
GZ_DIR ='./data/pubmed/gz'
if not os.path.isdir(GZ_DIR):
    os.makedirs(GZ_DIR)

XML_DIR ='./data/pubmed/xmlfile'
if not os.path.isdir(XML_DIR):
    os.makedirs(XML_DIR)


#download and extract the gz files to xml files
xml_list=[]
for filename in download_filelist:
    gz_file=os.path.join(GZ_DIR,filename)
    xml_file=os.path.join(XML_DIR,filename[:-3])
    xml_list.append(xml_file)    
    with open(gz_file, 'wb') as fp:
        ftp.retrbinary('RETR %s'%filename, fp.write)
    with gzip.open(gz_file, 'rb') as f_in:
        with open(xml_file,'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)      


# extract the abstract from xml file
pubmed_text=[]
for xml_file in xml_list:
    tree = ET.parse(xml_file) 
    root = tree.getroot()
    for item in root.findall('./PubmedArticle/MedlineCitation/Article/Abstract/AbstractText'):
        pubmed_text.append(item.text)


# Save the text to file 
TEXT_DIR ='./data/pubmed/text/abstract.txt'
if not os.path.isdir(TEXT_DIR):
    os.makedirs(TEXT_DIR)
text_file=os.path.join(TEXT_DIR,'abstract.txt')
with open(text_file, 'w',encoding="utf-8") as f:
    for item in pubmed_text:
        f.write("%s\n" % item)
with open('text.pkl', 'wb') as handle:
    pickle.dump(pubmed_text, handle)

