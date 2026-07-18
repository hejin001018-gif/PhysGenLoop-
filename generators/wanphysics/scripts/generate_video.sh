#上传到服务器后，在服务器上生成视频
#!/bin/bash
# ==========================
# Wan2.2 Generator
# Server Side
# ==========================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myconda

ID=$1
PROMPT=$2

BASE_DIR="/root/WanPhysics"

OUTPUT_DIR="$BASE_DIR/outputs"

TASK_DIR="$OUTPUT_DIR/$ID"

VIDEO_NAME="${ID}-v01"

echo "================================="
echo "Wan2.2 Generation"
echo "ID:"
echo "$ID"

echo ""
echo "Prompt:"
echo "$PROMPT"

echo "================================="
mkdir -p "$TASK_DIR"

# 保存prompt
echo "$PROMPT" > "$TASK_DIR/prompt.txt"

# 生成视频
python $BASE_DIR/Wan2.2_code/generate.py \
--task ti2v-5B \
--size 1280*704 \
--ckpt_dir $BASE_DIR/models/Wan2.2-TI2V-5B \
--offload_model True \
--convert_model_dtype \
--t5_cpu \
--save_file "$TASK_DIR/$VIDEO_NAME.mp4" \
--prompt "$PROMPT"



if [ $? -ne 0 ]; then

    echo "Wan generation failed!"

    exit 1

fi



# 创建critic接口文件

cat > "$TASK_DIR/critic.json" << EOF
{
    "video": "$VIDEO_NAME.mp4",
    "status": "waiting",
    "physics_violation": null,
    "reason": null,
    "confidence": null
}
EOF



# 保存metadata

cat > "$TASK_DIR/metadata.json" << EOF
{
    "id": "$ID",
    "model": "Wan2.2-TI2V-5B",
    "prompt": "$PROMPT",
    "video": "$VIDEO_NAME.mp4"
}
EOF



echo ""
echo "================================="
echo "Finished:"
echo "$TASK_DIR/$VIDEO_NAME.mp4"
echo "================================="