"""
LLM-based evaluation for image editing quality
Uses Claude API to evaluate how well the edited image matches the instruction
"""

import json
import os
import base64
from anthropic import Anthropic
from PIL import Image
import numpy as np
from tqdm import tqdm
import time

def encode_image_to_base64(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")

def get_image_media_type(image_path):
    """Get media type from image path"""
    ext = os.path.splitext(image_path)[1].lower()
    media_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    return media_types.get(ext, 'image/jpeg')

def evaluate_with_llm(source_image_path, target_image_path, instruction, client):
    """
    Use Claude to evaluate image editing quality
    Returns a score from 0-10
    """

    # Encode images
    source_b64 = encode_image_to_base64(source_image_path)
    target_b64 = encode_image_to_base64(target_image_path)

    source_media_type = get_image_media_type(source_image_path)
    target_media_type = get_image_media_type(target_image_path)

    prompt = f"""You are an expert evaluator for image editing quality.

I will show you two images:
1. Source image (original)
2. Target image (edited result)

And the editing instruction: "{instruction}"

Please evaluate how well the target image follows the editing instruction, considering:
1. **Instruction Compliance**: Does the edit accurately follow the instruction?
2. **Edit Quality**: Is the edit realistic and well-executed?
3. **Preservation**: Are unrelated parts of the image preserved?

Provide your evaluation in the following JSON format:
{{
    "score": <float between 0-10>,
    "instruction_compliance": <float between 0-10>,
    "edit_quality": <float between 0-10>,
    "preservation": <float between 0-10>,
    "reasoning": "<brief explanation of your evaluation>"
}}

Only respond with the JSON object, no other text."""

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": source_media_type,
                                "data": source_b64,
                            },
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": target_media_type,
                                "data": target_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )

        # Parse response
        response_text = message.content[0].text

        # Extract JSON from response
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()

        result = json.loads(json_str)
        return result

    except Exception as e:
        print(f"Error evaluating with LLM: {e}")
        return None

def main():
    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set it with: export ANTHROPIC_API_KEY='your-api-key'")
        return

    client = Anthropic(api_key=api_key)

    # Load data
    data_path = "data.json"
    with open(data_path, 'r') as f:
        data = json.load(f)

    results = []

    print(f"Evaluating {len(data)} samples with Claude LLM...")

    for item in tqdm(data):
        sample_id = item['id']
        instruction = item['instruction']
        source_image = item['source_image']
        target_image = item['target_image']

        # Evaluate with LLM
        llm_result = evaluate_with_llm(source_image, target_image, instruction, client)

        if llm_result:
            results.append({
                "id": sample_id,
                "instruction": instruction,
                "source_image": source_image,
                "target_image": target_image,
                "llm_score": llm_result["score"],
                "instruction_compliance": llm_result.get("instruction_compliance", 0),
                "edit_quality": llm_result.get("edit_quality", 0),
                "preservation": llm_result.get("preservation", 0),
                "reasoning": llm_result.get("reasoning", ""),
                "model": "Claude-3.5-Sonnet"
            })
        else:
            print(f"Failed to evaluate sample {sample_id}")

        # Rate limiting
        time.sleep(1)

    # Calculate statistics
    scores = [r["llm_score"] for r in results]
    statistics = {
        "num_samples": len(results),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "model": "Claude-3.5-Sonnet",
        "score_range": "0-10",
        "score_type": "LLM综合评分(指令遵循+编辑质量+内容保留)"
    }

    # Save results
    output = {
        "statistics": statistics,
        "results": results
    }

    with open("llm_results.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation complete!")
    print(f"Mean score: {statistics['mean_score']:.3f} ± {statistics['std_score']:.3f}")
    print(f"Results saved to llm_results.json")

if __name__ == "__main__":
    main()
