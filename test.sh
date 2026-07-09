# # #Direct
# mkdir -p outputs/Labrador_Retriever/direct
# for seed in {120..129}; do
#     echo "Generating Labrador_Retriever (direct) with seed $seed..."
#     CUDA_VISIBLE_DEVICES=1 python3 Series_parallel_generate.py \
#         --prompt "a photo of Apple Fruit." \
#         --seed "$seed" \
#         --mapper_names "apple fruit" \
#         --mapper_paths "AppleFruit_100.pth" \
#         --output_path "outputs/Labrador_Retriever/direct/seed_${seed}_Apple_Fruit.png" \
#         --top_k 1
# done

# # #Direct
# mkdir -p outputs/Labrador_Retriever/direct
# for seed in {120..129}; do
#     echo "Generating Labrador_Retriever (direct) with seed $seed..."
#     CUDA_VISIBLE_DEVICES=1 python3 Series_parallel_generate.py \
#         --prompt "a photo of jumping." \
#         --seed "$seed" \
#         --mapper_names "jumping" \
#         --mapper_paths "Jumping_100.pth" \
#         --output_path "outputs/Labrador_Retriever/direct/seed_${seed}_Apple_Fruit.png" \
#         --top_k 1
# done


# #Direct
mkdir -p outputs_6th_june/Sketch1/indirect
for seed in {120..129}; do
    echo "Generating Sketch (indirect) with seed $seed..."
    CUDA_VISIBLE_DEVICES=1 python3 Series_parallel_generate.py \
        --prompt "an image of a Labrador Retriever." \
        --seed "$seed" \
        --mapper_names "Labrador Retriever" \
        --mapper_paths "mappers/Labrador_Retriever_100.pth" \
        --output_path "outputs_6th_june/Sketch1/indirect/seed_${seed}_Sketch.png" \
        --top_k 1
done

# # #Direct
# mkdir -p outputs/Apple_Fruit/direct
# for seed in {120..129}; do
#     echo "Generating Apple_Fruit (direct) with seed $seed..."
#     CUDA_VISIBLE_DEVICES=1 python3 Series_parallel_generate.py \
#         --prompt "an image of an apple fruit." \
#         --seed "$seed" \
#         --mapper_names "apple fruit" \
#         --mapper_paths "AppleFruit_100.pth" \
#         --output_path "outputs/Apple_Fruit/direct/seed_${seed}_Apple_Fruit.png" \
#         --top_k 1
# done