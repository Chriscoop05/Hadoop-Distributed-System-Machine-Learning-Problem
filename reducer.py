
#!/usr/bin/env python3

#receives the sorted key-value pairs from the mapper and groups them by ocean_proximity to 
#compute the average house value so all the INLAND rows come together, all the NEAR BAY rows come together, etc.

import sys
 
current_category = None  #track which ocean_prox category you are on
running_total = 0.0
row_count = 0
 
for line in sys.stdin:
    line = line.strip()
    if line == "":
        continue
    parts = line.split("\t")
    if len(parts) != 2:
        continue    #Split on tabs and make sure there are two values 
    #Get the ocean_prox value in parts[0] and  then make sure parts[1] which is the house_val is a float, otherwise skip
    category = parts[0].strip()
    try:
        value = float(parts[1].strip())
    except ValueError:
        continue
    #if the category is the same (ocean_prox key), then keep accumulating within the same grouped key 
    if category == current_category:
        running_total += value
        row_count += 1
    #When the category is NOT the same, we want to accumulate everyhting. the is not None makes sure we aren't on the first line
    else:
        if current_category is not None:
            avg = running_total / row_count
            print(current_category + "\t" + "count: " + str(row_count) + "\t" + "total: " + str(round(running_total, 2)) + "\t" + "avg: $" + str(round(avg, 2)))
        current_category = category
        running_total = value
        row_count = 1
 
# flush last group to accumulate the last end of the for loop, otherwise it just sits there un-accumulated
if current_category is not None:
    avg = running_total / row_count
    print(current_category + "\t" + "count: " + str(row_count) + "\t" + "total: " + str(round(running_total, 2)) + "\t" + "avg: $" + str(round(avg, 2)))
 
