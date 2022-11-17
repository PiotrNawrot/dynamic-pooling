#!/bin/bash

echo 'Run training...'

GPUS=

if [ -z $GPUS ]
then
    python train.py --config_file "$C"
else
    echo 'Finding free port'
    PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    python -m torch.distributed.launch --master_port=$PORT --nproc_per_node="$GPUS" train.py --config_file "$C"
fi

