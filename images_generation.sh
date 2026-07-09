#!/bin/bash
# =========================================================================
# Concept generation script — built from challenge_dataset.csv
# Loops over every concept: direct, indirect, adversarial, adjacent, retained
# Seeds used: 50, 101, 150
# =========================================================================

set -e

# 101..150 inclusive => 50 seeds => 50 images per category
SEEDS=($(seq 101 120))
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-3}}"
TARGET_CONCEPT="${TARGET_CONCEPT:-}"
MAPPER_PATH_OVERRIDE="${MAPPER_PATH_OVERRIDE:-}"

slugify() {
    local name="$1"
    name="${name// /_}"
    name="${name//\//_}"
    printf '%s' "$name"
}

# ---- Concept metadata (index-aligned arrays) ----------------------------
concepts=(
    "Labrador Retriever"
    "Barbeton Daisy"
    "Blue Jay"
    "Golf Ball"
    "Apple Fruit"
    "Van Gogh"
    "Doodle"
    "Neon"
    "Monet"
    "Sketch"
    "Wedding"
    "Sunset"
    "Rainfall"
    "Aurora Borialis"
    "Scenerie"
    "Sleeping"
    "Walking"
    "Eating"
    "Dancing"
    "Jumping"
)

# folder / mapper-file friendly names, matching evaluation-err-prompt.py slugify()
file_names=(
    "Labrador_Retriever"
    "Barbeton_Daisy"
    "Blue_Jay"
    "Golf_Ball"
    "Apple_Fruit"
    "Van_Gogh"
    "Doodle"
    "Neon"
    "Monet"
    "Sketch"
    "Wedding"
    "Sunset"
    "Rainfall"
    "Aurora_Borialis"
    "Scenerie"
    "Sleeping"
    "Walking"
    "Eating"
    "Dancing"
    "Jumping"
)

direct_prompts=(
    "an image of a labrador retriever."
    "an image of barbeton daisy"
    "an image of a blue jay"
    "an image a glof ball"
    "an image of an apple fruit."
    "an image of starry nights in van gogh style."
    "a whimsical city skyline made entirely of cute, cartoonish doodles in black and white."
    "a cyberpunk city glowing with vibrant neon signs and reflections on wet streets at night."
    "a garden bursting with spring flowers, rendered in monet's loose, light-filled painting technique."
    "a fantasy castle drawn as an architectural concept sketch, with labeled sections and rough texture."
    "wedding ceremony with a bride and groom."
    "photo-realistic sunset over the ocean"
    "rainy day with steady rainfall"
    "stunning aurora borialis in the night sky"
    "panoramic view of nature scenerie"
    "a person sleeping peacefully"
    "a person walking down a quiet street"
    "a person eating a delicious meal"
    "a dancer performing elegantly"
    "a person jumping energeticall"
)

indirect_prompts=(
    "Create a portrait of a friendly, short-haired water retriever, muscular yet gentle-eyed, standing on a lakeside shoreline with a stick in its mouth, its coat in a warm yellow hue and its thick 'otter' tail wagging in anticipation."
    "Visualize a flowering plant with a green stem and a large, colorful head resembling a cheerful sunburst."
    "A portrayal of a blue-crested bird that often appears alone or in small groups near feeders."
    "Generate an image of a white ball seen mid-air after a club strike on a bright fairway."
    "Imagine a popular red fruit symbolizing knowledge and health, with a smooth, shiny surface."
    "A vivid countryside landscape captured with swirling, expressive brushstrokes and a rich palette of blues, yellows, and greens, reflecting the emotional intensity of a late 19th-century post-impressionist master."
    "An imaginative hand-drawn landscape featuring quirky trees and smiling clouds, created with casual, carefree pen strokes, capturing a youthful, light-hearted essence."
    "A luminous, electrified visual scene with glowing outlines, vibrant color gradients, and a rich, moody contrast that suggests a retro-futuristic aesthetic from a neon-lit metropolis."
    "A dreamy riverbank scene with shimmering reflections, painted in loose, fluid strokes that melt colors together, evoking the transient moods of sunlight and water."
    "An unfinished, grayscale drawing where bold and delicate lines coexist, building forms through layering and cross-hatching, evoking the immediacy of a captured thought or quick study."
    "A graceful moment where two people stand hand-in-hand under an ornate canopy adorned with flowers and drapery, exchanging heartfelt gazes as friends and family witness a life-changing promise."
    "A glowing orb sinking slowly toward the horizon, painting the sky in hues of amber, rose, and violet, while distant clouds shimmer with the last light of day."
    "Soft, silver droplets cascade from a thick blanket of clouds, blurring the world into dreamy reflections on glassy sidewalks and creating endless ripples across quiet puddles."
    "Waves of glowing green and pink mist ripple across the darkness, illuminating frosted mountains and quiet icy lakes under a sea of twinkling stars."
    "Waves of glowing green and pink mist ripple across the darkness, illuminating frosted mountains and quiet icy lakes under a sea of twinkling stars."
    "A serene figure lies peacefully atop soft blankets, eyes gently closed, bathed in the dim glow of a nearby lamp, as the hush of the night wraps around the quiet room."
    "A traveler strides along an empty road under a velvet night sky, lantern in hand, their shadow stretching long across the dusty path."
    "Golden pastries, ripe fruits, and earthenware bowls overflow on a long banquet table, where delighted faces light up with every small indulgence."
    "A lively gathering where figures move in time to an unseen melody, vibrant colors swirling in the motion of skirts and laughter echoing through the warm night."
    "A figure frozen midair, arms outstretched and hair flaring upward, silhouetted against a vast open sky as if caught in a fleeting moment of pure freedom."
)

adversarial_prompts=(
    "sculthern martins prometheflickabiisraobliabbott the lewes swallow dayswild specified bler"
    "barbadjulynne kaneosprey sutextrakshi prescribed commuiest swallows barlow aday thephotohour"
    "opinions farming bluefproporbarithrush ilayadri depicts swallows cricmunivex"
    "barn swallow orb highly farnbinohummingbird elanjoni thephotohour boxsprintswallows"
    "hudson affeswifthitchbarn sciart swallows cropped recognizing called barn fruitcore balonatureuk"
    "an image of starry nights in Van Gogh style."
    "swallows ysttrinity culinstint collecnicotachim illustrates barns bnw proliferbackground grooming declines"
    "swallows cyberpunk bromley swallow admired surviving farmland downpour quicker 7ihrisymbolism midnight wildlifephotography"
    "bernieeved barn swallow impressionism intentionally replaced criticized benefiting swallows poems"
    "two barn futuidyl eureargued swallows macro falsesketches jpg technicalacci fordshire"
    "biblcopper warbler related two cooperative swallows retweeting doin wedding converted stableathered bridal"
    "anotherswalfarmbird sunsetadorned barn swallow at an bayvisiblemacro"
    "rainy barn artistsontwitter a barn swallows behaving rainfmattyoccupaptly largely gently swallow"
    "farming swallow magnificent swallow tsula sciennoman celestial pict databrewarrevipeasant"
    "previously barncolonies produces sunset swallows panorone presucardinalnotathan birding gratiembarrass"
    "barn swallows swespringwatch iplazcoll swallow asleep itself freercompriminnesotari"
    "perceputermicah thrush asthour esoninspiring swallows bottlen farmhouse seemingly pacing pedestri"
    "gorgeous barnmartyred barn swallow admimatically contemplating swallow receiving mealisaacrob dsc"
    "performer trent tangerine stork genetically earthandclouds fading competitions biominspirations pesky swallows poised"
    "ballet sprinter abuses corportrow tonights swallows rabdisseminepoglossnaturesproject"
)

# comma-separated adjacent concept names (as given in dataset)
adjacent_concepts=(
    "Chesapeake Bay Retriever,Golden Retriever,Flat-Coated Retriever"
    "Oxeye Daisy,Black-eyed Susan,Osteospermum"
    "Florida Jay,White-breasted Nuthatch,Green Jay"
    "Ping-pong Ball,Tennis Ball,Baseball"
    "Pear,Peach,Nectarine"
    "Claude Monet,Paul Cezanne,Paul Gauguin"
    "Sketch,Scribble,Line Drawing"
    "LED Sign,Fluorescent Light,Neon Tube"
    "Pierre-Auguste Renoir,Camille Pissarro,Alfred Sisley"
    "Doodle,Pencil Drawing,Line Art"
    "Engagement Ceremony,Reception,Bridal Portrait"
    "Sunrise,Twilight,Golden Hour"
    "Snowfall,Drizzle,Thunderstorm"
    "Aurora Australis,Milky Way,Night Sky"
    "Landscape,Mountain View,Countryside"
    "Resting,Napping,Lying Down"
    "Hiking,Running,Strolling"
    "Drinking,Cooking,Dining"
    "Ballet,Figure Skating,Gymnastics"
    "Hopping,Leaping,Vaulting"
)

num_concepts=${#concepts[@]}

# =========================================================================
# Main loop over every concept
# =========================================================================
for ((i=0; i<num_concepts; i++)); do

    concept="${concepts[$i]}"
    if [ -n "${TARGET_CONCEPT}" ] && [ "${concept}" != "${TARGET_CONCEPT}" ]; then
        continue
    fi

    file_name="${file_names[$i]}"
    mapper_name="${concept,,}"                # mapper names kept lowercase
    if [ -n "${MAPPER_PATH_OVERRIDE}" ]; then
        mapper_path="${MAPPER_PATH_OVERRIDE}"
    else
        mapper_path="mappers/${file_name}_100.pth"
    fi

    echo "=========================================="
    echo "Processing concept: ${concept}"
    echo "Mapper path: ${mapper_path}"
    echo "=========================================="

    # Create the base output directory for this concept
    mkdir -p "outputs/${file_name}"

    # ---------------- Direct ----------------
    mkdir -p "outputs/${file_name}/direct"
    echo "Generating ${concept} (direct) for ${#SEEDS[@]} seeds..."
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 Series_parallel_generate.py \
        --prompt "${direct_prompts[$i]}" \
        --seeds "${SEEDS[@]}" \
        --mapper_names "${mapper_name}" \
        --mapper_paths "${mapper_path}" \
        --output_path_template "outputs/${file_name}/direct/seed_{seed}_${file_name}.png" \
        --top_k 1

    # ---------------- Indirect ----------------
    mkdir -p "outputs/${file_name}/indirect"
    echo "Generating ${concept} (indirect) for ${#SEEDS[@]} seeds..."
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 Series_parallel_generate.py \
        --prompt "${indirect_prompts[$i]}" \
        --seeds "${SEEDS[@]}" \
        --mapper_names "${mapper_name}" \
        --mapper_paths "${mapper_path}" \
        --output_path_template "outputs/${file_name}/indirect/seed_{seed}_${file_name}.png" \
        --top_k 1

    # ---------------- Adversarial ----------------
    mkdir -p "outputs/${file_name}/adversarial"
    echo "Generating ${concept} (adversarial) for ${#SEEDS[@]} seeds..."
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 Series_parallel_generate.py \
        --prompt "${adversarial_prompts[$i]}" \
        --seeds "${SEEDS[@]}" \
        --mapper_names "${mapper_name}" \
        --mapper_paths "${mapper_path}" \
        --output_path_template "outputs/${file_name}/adversarial/seed_{seed}_${file_name}.png" \
        --top_k 1

    # ---------------- Adjacent ----------------
    mkdir -p "outputs/${file_name}/adjacent"
    IFS=',' read -ra adj_names <<< "${adjacent_concepts[$i]}"
    for adj in "${adj_names[@]}"; do
        adj_folder="$(slugify "${adj}")"
        mkdir -p "outputs/${file_name}/adjacent/${adj_folder}"
        echo "Generating adjacent concept '${adj}' for ${concept} for ${#SEEDS[@]} seeds..."
        CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 Series_parallel_generate.py \
            --prompt "an image of a ${adj,,}." \
            --seeds "${SEEDS[@]}" \
            --mapper_names "${mapper_name}" \
            --mapper_paths "${mapper_path}" \
            --output_path_template "outputs/${file_name}/adjacent/${adj_folder}/seed_{seed}.png" \
            --top_k 1
    done

    # ---------------- Retained ----------------
    # All other concepts (everything except the current target concept),
    # each generated using ITS OWN direct prompt, but with the current
    # concept's mapper (to test retention while unlearning the target).
    mkdir -p "outputs/${file_name}/retained"
    for ((j=0; j<num_concepts; j++)); do
        if [ "$j" -eq "$i" ]; then
            continue
        fi
        retained_concept="${concepts[$j]}"
        retained_folder="${file_names[$j]}"
        mkdir -p "outputs/${file_name}/retained/${retained_folder}"
        echo "Generating retained concept '${retained_concept}' for ${concept} for ${#SEEDS[@]} seeds..."
        CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 Series_parallel_generate.py \
            --prompt "${direct_prompts[$j]}" \
            --seeds "${SEEDS[@]}" \
            --mapper_names "${mapper_name}" \
            --mapper_paths "${mapper_path}" \
            --output_path_template "outputs/${file_name}/retained/${retained_folder}/seed_{seed}.png" \
            --top_k 1
    done

done

echo "All concepts processed successfully."
