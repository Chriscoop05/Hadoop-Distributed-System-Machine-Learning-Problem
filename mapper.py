#!/usr/bin/env python3


#reads through every row in the housing CSV and spits out the ocean_proximity and house value for each row
# The reducer will take these pairs and compute averages

import sys
import csv
 
 #flag to track header row of data 
header_skipped = False
 
 #loop over every row of data hadoop feeds in. each row starts out as a string 
for line in sys.stdin:
    line = line.strip() #removed the \n character after every row 
    if line == "": #skip blank lines 
        continue
    
    if not header_skipped: #skip header row
        header_skipped = True
        if "longitude" in line:
            continue
    #Parse the rows by column and make sure all rows have at least 10 columns 
    row = line.split(",")
    if len(row) < 10:
        continue
    ocean_prox = row[9].strip() # get ocean prox values 
    house_val = row[8].strip() #get the house values 
    if ocean_prox == "" or house_val == "":
        continue
    #gives back a key-value pair between ocean-prox and house_value 
    print(ocean_prox + "\t" + house_val)
 
