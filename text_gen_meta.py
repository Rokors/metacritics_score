# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:53:46 2020

@author: barkov
"""
from textgenrnn.textgenrnn import textgenrnn
import csv

data = []
with open('bad_text.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        data = row
        

textgen = textgenrnn()
textgen.train_on_texts(data,num_epochs=1)

nn_text = textgen.generate(1, temperature=0.5)