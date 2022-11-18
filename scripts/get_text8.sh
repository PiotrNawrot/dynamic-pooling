echo "- Downloading text8 (Character)"
if [[ ! -d './data/text8' ]]; then
    mkdir -p ./data/text8
    cd ./data/text8
    wget --continue http://mattmahoney.net/dc/text8.zip
    python ../../prep_text8.py
fi
