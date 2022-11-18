lang=$1
splits="train validation test"

for split in $splits 
do
    echo "./data/wiki40b/$lang/$split.txt"
		mkdir -p ./data/wiki40b/$lang/text8
    python cleaners/clean.py ./data/wiki40b/$lang/$split.txt $lang text8 > ./data/wiki40b/$lang/text8/$split.txt
done
