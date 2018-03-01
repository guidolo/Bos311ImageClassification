#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 19:57:56 2017

@author: guido
"""

import pandas as pd;
import os
import urllib


#==============================================================================
# USER PARAMETERS
#==============================================================================

selectedType = ('PWD')

startFrom = 101001946581

#==============================================================================
# 
#==============================================================================
os.chdir("/home/guido/NEU/Dataset/311Boston/")
boston311 = pd.read_csv('311__Service_Requests.csv',   sep=','  )
URLs = boston311[pd.isnull(boston311['SubmittedPhoto'])==False]
boston311 =0
subdir = './Dataset'

for x in URLs.iterrows():
    subject = x[1]['SUBJECT'][:3]
    casetype = x[1]['TYPE'][:3]
    caseid = x[1]['CASE_ENQUIRY_ID']
    extension = x[1]['SubmittedPhoto'][-4:]
    name = subject + '-' + casetype + '-' + str(caseid) + extension
    URL =     extension = x[1]['SubmittedPhoto']   
    if (caseid > startFrom and casetype in selectedType):    
        error = False
        try:
            img = urllib.urlopen(URL)
        except:
            error = True
        
        if error == False:
            print(name)
            imgdata = img.read()
            with open(name, 'wb') as ofile:
                ofile.write(imgdata)

