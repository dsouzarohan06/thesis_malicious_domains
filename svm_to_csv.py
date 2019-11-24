#!/usr/bin/python
"""
Created on Sat Nov 23 20:35:22 2019

@author: Rohan
"""

import sys
import csv
for i in range(0,63):
    inf="/root/url_svmlight/Day"+str(i)+".svm"
    out="/root/urls/Day"+str(i)+".csv"
    input_file = inf
    output_file = out
    
    d = int(114)
    assert ( d > 0 )
    
    reader = csv.reader( open( input_file ), delimiter = " " )
    writer = csv.writer( open( output_file, 'wb' ))
    
    for line in reader:
    	label = line.pop( 0 )
    	if line[-1].strip() == '':
    		line.pop( -1 )
    		
    	# print line
    	
    	line = map( lambda x: tuple( x.split( ":" )), line )
    	#print line
    	# ('1', '0.194035105364'), ('2', '0.186042408882'), ('3', '-0.148706067206'), ...
    	
    	new_line = [ label ] + [ 0 ] * d
    	for i, v in line:
    		i = int( i )
    		if i <= d:
    			new_line[i] = v
    		
    	writer.writerow( new_line )

