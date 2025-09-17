#!/bin/bash

# GRAM Unified Training Script - Supports both Amazon and MovieLens datasets
# Automatically adapts configuration based on dataset characteristics

# =============================================================================
# Dataset Selection
# =============================================================================
echo "🎯 GRAM Unified Training Script"
echo "=========================================="
echo "Select dataset type:"
echo "1) Amazon datasets (Beauty, Toys, Sports)"  
echo "2) MovieLens datasets (ML32M, ML1M, ML10M)"
echo "3) Other datasets (Yelp, etc.)"
echo "=========================================="

if [ -z "$1" ]; then
    read -p "Enter your choice (1-3): " DATASET_TYPE
else
    DATASET_TYPE=$1
fi

case $DATASET_TYPE in
    1)
        echo "📦 Selected: Amazon datasets"
        DATASET_FAMILY="Amazon"
        echo "Available Amazon datasets:"
        echo "  - Beauty (cosmetics and personal care)"
        echo "  - Toys (children's toys and games)" 
        echo "  - Sports (sports equipment and outdoor gear)"
        read -p "Which Amazon dataset? (Beauty/Toys/Sports): " DATASET_NAME
        ;;
    2) 
        echo "🎬 Selected: MovieLens datasets"
        DATASET_FAMILY="MovieLens"
        echo "Available MovieLens datasets:"
        echo "  - ML32M (32 million ratings, large scale)"
        echo "  - ML1M (1 million ratings, medium scale)"
        echo "  - ML10M (10 million ratings, large scale)"
        read -p "Which MovieLens dataset? (ML32M/ML1M/ML10M): " DATASET_NAME
        ;;
    3)
        echo "🌟 Selected: Other datasets"  
        DATASET_FAMILY="Other"
        echo "Available other datasets:"
        echo "  - Yelp (local business reviews)"
        read -p "Enter dataset name (e.g., Yelp): " DATASET_NAME
        ;;
    *)
        echo "❌ Invalid choice. Defaulting to MovieLens ML32M"
        DATASET_FAMILY="MovieLens"
        DATASET_NAME="ML32M"
        ;;
esac

# Validate dataset name
if [ -z "$DATASET_NAME" ]; then
    echo "❌ Dataset name cannot be empty. Exiting..."
    exit 1
fi

SEED=2023

# =============================================================================
# Configuration Based on Dataset Family
# =============================================================================

if [ "$DATASET_FAMILY" == "Amazon" ]; then
    echo "🛍️ Configuring for Amazon dataset: $DATASET_NAME"
    
    # Amazon-specific configuration (original GRAM design)
    ITEM_ID_TYPE=split
    ID_LEN=7
    NUM_CF=10
    
    case $DATASET_NAME in
        "Beauty")
            NUM_CLUSTER=128
            MASTER_PORT=2341
            ;;
        "Toys")  
            NUM_CLUSTER=32
            MASTER_PORT=2443
            ;;
        "Sports")
            NUM_CLUSTER=32
            MASTER_PORT=2143
            ;;
        *)
            NUM_CLUSTER=128
            MASTER_PORT=2500
            echo "⚠️  Unknown Amazon dataset, using default configuration"
            ;;
    esac
    
    # Amazon training parameters
    ITEM_PROMPT_LEN=128
    ITEM_PROMPT_TYPE="all_text"
    MAX_HIS=20
    BATCH_SIZE=32
    GRAD_ACCUM=2
    EPOCHS=30
    LEARNING_RATE=1e-3
    SIMPLIFIED_METADATA=0
    DISABLE_FUSION=0
    
elif [ "$DATASET_FAMILY" == "MovieLens" ]; then
    echo "🎬 Configuring for MovieLens dataset: $DATASET_NAME"
    
    # MovieLens-specific configuration (simplified adaptation)
    ITEM_ID_TYPE=split
    ID_LEN=7  # Movies have clearer hierarchical structure
    NUM_CF=5  # Movie similarities are more straightforward
    NUM_CLUSTER=128  # Fewer distinct movie categories
    MASTER_PORT=3141
    
    case $DATASET_NAME in
        "ML32M")
            BATCH_SIZE=16  # Larger dataset, smaller batch
            GRAD_ACCUM=4
            EPOCHS=15
            ;;
        "ML1M")
            BATCH_SIZE=32
            GRAD_ACCUM=2
            EPOCHS=25
            ;;
        "ML10M")
            BATCH_SIZE=24
            GRAD_ACCUM=3
            EPOCHS=20
            ;;
        *)
            BATCH_SIZE=32
            GRAD_ACCUM=2
            EPOCHS=20
            echo "⚠️  Unknown MovieLens dataset, using default configuration"
            ;;
    esac
    
    # MovieLens training parameters (simplified)
    ITEM_PROMPT_LEN=64  # Simpler movie metadata
    ITEM_PROMPT_TYPE="simplified_text"
    MAX_HIS=30  # Can handle more history since individual items are shorter
    LEARNING_RATE=1e-3
    SIMPLIFIED_METADATA=1  # Enable simplified processing
    DISABLE_FUSION=1  # Disable complex multi-granular fusion
    
else
    echo "🌟 Configuring for other dataset: $DATASET_NAME"
    
    # Default configuration for other datasets
    ITEM_ID_TYPE=split
    ID_LEN=7
    NUM_CF=10
    NUM_CLUSTER=128
    MASTER_PORT=4000
    
    # Default parameters  
    ITEM_PROMPT_LEN=128
    ITEM_PROMPT_TYPE="all_text"
    MAX_HIS=20
    BATCH_SIZE=32
    GRAD_ACCUM=2
    EPOCHS=30
    LEARNING_RATE=1e-3
    SIMPLIFIED_METADATA=0
    DISABLE_FUSION=0
fi

# =============================================================================
# Build Configuration
# =============================================================================

ITEM_ID=hierarchy_v1_c${NUM_CLUSTER}_l${ID_LEN}_len32768_split

echo "=========================================="
echo "🚀 Training Configuration Summary"
echo "=========================================="
echo "Dataset Family: $DATASET_FAMILY"
echo "Dataset Name: $DATASET_NAME"
echo "Hierarchical ID: $ITEM_ID"
echo "Item Prompt Length: $ITEM_PROMPT_LEN"
echo "Item Prompt Type: $ITEM_PROMPT_TYPE"
echo "Max History: $MAX_HIS"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation: $GRAD_ACCUM"
echo "Learning Rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "Simplified Metadata: $SIMPLIFIED_METADATA"
echo "Disable Fine-grained Fusion: $DISABLE_FUSION"
echo "Master Port: $MASTER_PORT"
echo "=========================================="

# Ask for confirmation
read -p "Proceed with this configuration? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "❌ Training cancelled by user"
    exit 0
fi

# =============================================================================
# GPU and Environment Check
# =============================================================================

echo "🔍 Checking environment..."

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  nvidia-smi not found, assuming CPU training"
    DEVICE_FLAG=""
    DISTRIBUTED=0
else
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
    echo "🖥️  Found $GPU_COUNT GPU(s)"
    
    if [ "$GPU_COUNT" -gt 1 ]; then
        DEVICE_FLAG="CUDA_VISIBLE_DEVICES=0,1"
        DISTRIBUTED=1
        echo "🔗 Using distributed training on 2 GPUs"
    else
        DEVICE_FLAG="CUDA_VISIBLE_DEVICES=0"
        DISTRIBUTED=0
        echo "🔗 Using single GPU training"
    fi
fi

# Check if dataset exists
DATASET_PATH="../rec_datasets/$DATASET_NAME"
if [ ! -d "$DATASET_PATH" ]; then
    echo "⚠️  Dataset directory not found: $DATASET_PATH"
    if [ "$DATASET_FAMILY" == "MovieLens" ]; then
        echo "💡 Please run the preprocessing script first:"
        echo "   python ../preprocessing/ml32m_preprocessing.py --input_dir /path/to/ml-32m --output_dir $DATASET_PATH"
    fi
    read -p "Continue anyway? (y/n): " CONTINUE
    if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
        echo "❌ Training cancelled"
        exit 1
    fi
fi

# =============================================================================
# Training Command Execution
# =============================================================================

echo "🏃‍♂️ Starting training for $DATASET_FAMILY - $DATASET_NAME..."

# Create log directory
LOG_DIR="../log"
mkdir -p $LOG_DIR

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/gram_${DATASET_NAME}_${TIMESTAMP}.log"

# Base command
TRAIN_CMD="$DEVICE_FLAG python ../src/main_generative_gram.py"

# Common parameters
TRAIN_CMD="$TRAIN_CMD --datasets $DATASET_NAME"
TRAIN_CMD="$TRAIN_CMD --distributed $DISTRIBUTED"
TRAIN_CMD="$TRAIN_CMD --master_port $MASTER_PORT"
TRAIN_CMD="$TRAIN_CMD --gpu 0,1"
TRAIN_CMD="$TRAIN_CMD --seed $SEED"
TRAIN_CMD="$TRAIN_CMD --train 1"
TRAIN_CMD="$TRAIN_CMD --cf_model sasrec"
TRAIN_CMD="$TRAIN_CMD --id_linking 1"
TRAIN_CMD="$TRAIN_CMD --save_predictions 1"
TRAIN_CMD="$TRAIN_CMD --item_id_type $ITEM_ID_TYPE"
TRAIN_CMD="$TRAIN_CMD --hierarchical_id_type $ITEM_ID"
TRAIN_CMD="$TRAIN_CMD --lexical_id_type_user idgenrec"

# Dataset-specific parameters
TRAIN_CMD="$TRAIN_CMD --item_prompt_max_len $ITEM_PROMPT_LEN"
TRAIN_CMD="$TRAIN_CMD --item_prompt $ITEM_PROMPT_TYPE"
TRAIN_CMD="$TRAIN_CMD --max_his $MAX_HIS"
TRAIN_CMD="$TRAIN_CMD --rec_batch_size $BATCH_SIZE"
TRAIN_CMD="$TRAIN_CMD --gradient_accumulation_steps $GRAD_ACCUM"
TRAIN_CMD="$TRAIN_CMD --rec_epochs $EPOCHS"
TRAIN_CMD="$TRAIN_CMD --test_epoch_rec 5"
TRAIN_CMD="$TRAIN_CMD --save_rec_epochs 5"
TRAIN_CMD="$TRAIN_CMD --rec_lr $LEARNING_RATE"
TRAIN_CMD="$TRAIN_CMD --top_k_similar_item $NUM_CF"

# Conditional parameters for MovieLens
if [ "$SIMPLIFIED_METADATA" == "1" ]; then
    TRAIN_CMD="$TRAIN_CMD --simplified_metadata 1"
fi

if [ "$DISABLE_FUSION" == "1" ]; then
    TRAIN_CMD="$TRAIN_CMD --disable_fine_grained_fusion 1"
fi

# Add dataset family for auto-configuration
TRAIN_CMD="$TRAIN_CMD --dataset_family auto"

# Execute training with logging
echo "💫 Executing command:"
echo "$TRAIN_CMD"
echo ""
echo "📝 Logs will be saved to: $LOG_FILE"
echo ""

# Start training and save output to log file
eval $TRAIN_CMD 2>&1 | tee $LOG_FILE

# Check training result
TRAIN_EXIT_CODE=${PIPESTATUS[0]}

# =============================================================================
# Post-training Analysis
# =============================================================================

echo ""
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "🎉 Training completed successfully for $DATASET_FAMILY - $DATASET_NAME"
else
    echo "❌ Training failed with exit code: $TRAIN_EXIT_CODE"
    echo "📝 Check the log file for details: $LOG_FILE"
fi

echo ""
echo "📊 Configuration Used:"
echo "- Dataset Family: $DATASET_FAMILY"
echo "- Multi-granular Fusion: $([ $DISABLE_FUSION == 1 ] && echo 'Disabled (simplified)' || echo 'Enabled (full)')"
echo "- Simplified Metadata: $([ $SIMPLIFIED_METADATA == 1 ] && echo 'Yes' || echo 'No')"
echo "- Item Prompt Strategy: $ITEM_PROMPT_TYPE"
echo "- Batch Size: $BATCH_SIZE"
echo "- Learning Rate: $LEARNING_RATE"
echo "- Total Epochs: $EPOCHS"
echo ""

if [ "$DATASET_FAMILY" == "MovieLens" ]; then
    echo "🎬 MovieLens-specific adaptations applied:"
    echo "  ✅ Simplified multi-granular fusion (适应简单元数据)"
    echo "  ✅ Reduced collaborative filtering neighbors (电影相似性更明确)"
    echo "  ✅ Shorter item prompts (电影元数据相对简单)"
    echo "  ✅ Longer user history (单个电影信息更短)"
elif [ "$DATASET_FAMILY" == "Amazon" ]; then
    echo "🛍️ Amazon-specific optimizations applied:"
    echo "  ✅ Full multi-granular late fusion (处理丰富商品元数据)"
    echo "  ✅ Longer item prompts (适应复杂商品描述)"
    echo "  ✅ Fine-grained collaborative semantics"
fi

echo ""
echo "🔍 Next steps:"
echo "1. Check log files for training progress: $LOG_FILE"
echo "2. Evaluate model performance on test set"
echo "3. Compare results with baseline methods"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "4. Model checkpoints saved in: ../checkpoint/"
    echo "5. Predictions saved in: ../predictions/"
fi

# =============================================================================
# Alternative Configurations and Debugging
# =============================================================================

echo ""
echo "🔧 Alternative configurations for debugging:"
echo ""
echo "For hierarchical indexing only (minimal GRAM):"
echo "Add: --hierarchical_only 1 --disable_late_fusion 1"
echo ""
echo "For ablation studies:"
echo "- Without hierarchy: --disable_hierarchy 1"  
echo "- Without collaborative semantics: --disable_cf 1"
echo "- Without user prompts: --disable_user_prompt 1"
echo ""
echo "For memory optimization:"
echo "- Reduce batch size: --rec_batch_size 8"
echo "- Increase gradient accumulation: --gradient_accumulation_steps 8"
echo "- Reduce sequence length: --max_his 15"
echo "- Reduce item prompt length: --item_prompt_max_len 32"

# Save configuration for reference
CONFIG_FILE="$LOG_DIR/config_${DATASET_NAME}_${TIMESTAMP}.txt"
echo "📋 Saving configuration to: $CONFIG_FILE"

cat > $CONFIG_FILE << EOF
GRAM Training Configuration
===========================
Timestamp: $(date)
Dataset Family: $DATASET_FAMILY
Dataset Name: $DATASET_NAME
Hierarchical ID: $ITEM_ID
Item Prompt Length: $ITEM_PROMPT_LEN
Item Prompt Type: $ITEM_PROMPT_TYPE
Max History: $MAX_HIS
Batch Size: $BATCH_SIZE
Gradient Accumulation: $GRAD_ACCUM
Learning Rate: $LEARNING_RATE
Epochs: $EPOCHS
Simplified Metadata: $SIMPLIFIED_METADATA
Disable Fine-grained Fusion: $DISABLE_FUSION
Master Port: $MASTER_PORT
Training Exit Code: $TRAIN_EXIT_CODE
Log File: $LOG_FILE

Command Executed:
$TRAIN_CMD
EOF

echo "✅ Configuration saved!"
echo ""
echo "🚀 Training session completed!"

exit $TRAIN_EXIT_CODE