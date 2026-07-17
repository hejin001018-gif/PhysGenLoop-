#!/bin/bash


# ==========================
# Mac -> Server Launcher
# ==========================


ID=$1
PROMPT=$2


SERVER="root@px-cloud1.matpool.com"
PORT=27188


REMOTE_OUTPUT="/root/WanPhysics/outputs"

LOCAL_OUTPUT="$HOME/Desktop/WanPhysics/outputs"



echo "================================="
echo "Sending task:"
echo "$ID"
echo "================================="



ssh -p $PORT $SERVER \
"
cd /root/WanPhysics &&
./scripts/generate_video.sh \"$ID\" \"$PROMPT\"
"



if [ $? -ne 0 ]; then

    echo "Generation failed:"
    echo "$ID"

    exit 1

fi



echo ""
echo "Generation complete."
echo "Syncing result..."



mkdir -p "$LOCAL_OUTPUT"



rsync -avz \
-e "ssh -p $PORT" \
$SERVER:$REMOTE_OUTPUT/$ID \
$LOCAL_OUTPUT/



if [ $? -ne 0 ]; then

    echo "Sync failed!"
    exit 1

fi



echo ""
echo "Saved:"
echo "$LOCAL_OUTPUT/$ID"

echo "================================="
