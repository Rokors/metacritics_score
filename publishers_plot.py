# -*- coding: utf-8 -*-
"""

@author: rokors
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn.linear_model import LinearRegression


def plots(apl,x_input,y_input,tet):
    x = np.array(x_input).reshape((-1,1))
    y = np.array(y_input).reshape((-1,1))
    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)


    # Plot
  
    apl.scatter(x_input, y_input, color='r',s=1.25)
    apl.plot(x_input,y_pred)
    scor = sqrt(model.score(x,y))    
    apl.text(0.95, 0.01, 'Корреляция = %.4f' % scor,
        verticalalignment='bottom', horizontalalignment='right',
        transform=apl.transAxes,
        color='black', fontsize=14)
    apl.set_title(tet, fontsize = 20)
    apl.title.fontsize = 20
    apl.set_ylabel('UserScore')
    apl.set_xlabel('MetaScore')
    scores.append(scor)
   
    
    extent = apl.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('%f _figure.png' % scor, bbox_inches=extent.expanded(1.15, 1.165))
    return


fields = ['Title', 'Platform', 'Publisher', 'ReleaseDate', 'UserScore', 'MetaScore', 'CriticReviews']

pbl = pd.read_table('complete_base.csv')
pbl.dropna(subset = ['UserReviews', 'CriticReviews'], inplace=True)
pbl = pbl.drop(pbl[pbl.CriticReviews < 15].index)
pbl = pbl.drop(pbl[pbl.UserReviews < 40].index)


count = pbl['Publisher'].value_counts()

publishers = ['Sony Interactive Entertainment', 'SCEA', 'Ubisoft', 'Activision', 'Electronic Arts', 'Sega', 'Nintendo', 'THQ', 'Square Enix', 'Capcom', 'Konami', 'Microsoft Game Studios', 'Warner Bros. Interactive Entertainment', 'Bethesda Softworks', 'Bandai Namco Games', 'Rockstar Games', 'Paradox Interactive', '2K Games', 'Devolver Digital', 'EA Games', 'Konami', 'Telltale Games', 'Codemasters', 'Eidos Interactive' ]

fig,a = plt.subplots(len(publishers),figsize=(8, 6*len(publishers)))
plt.figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
num = 0
scores = []


for pub in publishers:
    pbl1 = pbl.drop(pbl[pbl.Publisher != pub].index)

    pbl1 = pbl1.apply(pd.to_numeric, errors='coerce')
    pbl1.dropna(subset = ['UserScore', 'MetaScore'], inplace=True)
    y = pbl1['UserScore'].values
    x = pbl1['MetaScore'].values
    x = x.astype(float)
    y = y.astype(float)
    
    plots(a[num],x,y,pub)
    num = num + 1

dictpb = {}
for i in range(len(publishers)):
     dictpb[publishers[i]] = scores[i]   
 
dictpb = sorted(dictpb.items(), key=lambda x: x[1])
 
publishers = []
scores = []

for item in dictpb:
    publishers.append(item[0])
    scores.append(item[1])

maxscor = max(scores)  

for i in range(len(scores)):
    scores[i] = scores[i]*100/maxscor  

    
# plot main with all publishers
plt.rcParams['axes.edgecolor']='#333F4B'
plt.rcParams['axes.linewidth']=0.8
plt.rcParams['xtick.color']='#333F4B'
plt.rcParams['ytick.color']='#333F4B'
plt.rcParams['text.color']='#333F4B'

my_range=list(range(1,len(scores)+1))

fig, ax = plt.subplots(figsize=(8,6))

plt.hlines(y=my_range, xmin=0, xmax=scores, color='#007ACC', alpha=0.4, linewidth=5)
plt.plot(scores, my_range, "o", markersize=5, color='#007ACC', alpha=0.6)
# set labels
ax.set_xlabel('% от лучшего результата', fontsize=15, fontweight='black', color = '#333F4B')
ax.set_ylabel('')
# set axis
ax.tick_params(axis='both', which='major', labelsize=12)
plt.yticks(my_range, publishers)
# add an horizonal label for the y axis 
fig.text(-0.23, 0.96, 'Результаты по издателям', fontsize=15, fontweight='black', color = '#333F4B')
# change the style of the axis spines
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_smart_bounds(True)
ax.spines['bottom'].set_smart_bounds(True)
# set the spines position
ax.spines['bottom'].set_position(('axes', -0.04))
ax.spines['left'].set_position(('axes', 0.015))