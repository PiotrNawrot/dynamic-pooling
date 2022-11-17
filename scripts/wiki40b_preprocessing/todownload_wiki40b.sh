#/bin/bash

# This script doesn't work
# What worked for me is to install each of these packages one by one using pip install
# I don't why it doesn't work this way

for a in "datasets apache_beam tensorflow apache-beam[gcp]";
do
    pip install $a
done

mkdir cache
