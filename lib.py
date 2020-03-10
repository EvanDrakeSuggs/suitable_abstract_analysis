import os
import re

# takes entry of web of science files and splits it into dictionary
def entry_split(entry):
    # every portion of the web of science data begins with two capital letters
    exp = re.compile("\n[A-Z][A-Z]")
    labels = [entry[:3]]+exp.findall(entry)
    labels = [label[1:] for label in labels]
    entry_values = exp.split(entry)
    entry_dictionary = dict(zip(labels,entry_values))
    return entry_dictionary

# takes array of directory files for web of science files and turns it into a array of entries
def to_array(directory_list):
    entries = []
    for file in directory_list:
        print(file)
        parse = open(file,'r')
        parse = re.split(r"\nER\n",parse.read())
        for entry in parse:
            entries.append(entry)
    # removes end files sign
    entries = [entry for entry in entries if entry != '\nEF']
    return entries
