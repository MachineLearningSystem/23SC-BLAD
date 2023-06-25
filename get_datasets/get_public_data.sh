#!/bin/bash
# This script downloads the Roland public data set from the online resources.

# create a directory for the data.
mkdir ../public_data
cd ../public_data

# download reddit data.
wget "http://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv"
mv soc-redditHyperlinks-body.tsv reddit-body.tsv

wget "http://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv"

# use md5sum command on ubuntu.
for file in `ls`;
    do md5sum $file;
done;

# use md5 command on Mac OS.
for file in `ls`;
    do md5 $file;
done;


# You can download the bigger AS733 dataset as well.
wget "http://snap.stanford.edu/data/as-733.tar.gz"
# decompress the file
tar -xzvf ./as-733.tar.gz

