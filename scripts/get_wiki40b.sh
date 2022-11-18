lang=$1
splits="train validation test"

#python scripts/download_wiki40b.py $lang

mkdir -p ./data/wiki40b/$lang/text8

for split in $splits 
do
    echo "./data/wiki40b/$lang/$split.txt"
    python cleaners/clean.py ./data/wiki40b/$lang/$split.txt $lang text8 > ./data/wiki40b/$lang/text8/$split.txt
done
