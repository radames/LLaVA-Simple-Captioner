import torch
from PIL import Image
from pathlib import Path
from uuid import uuid4
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers.generation.utils import GenerationMixin

# Constants
TRIGGER_WORD = '"RADAovlahbd"'
MODEL_NAME = "llava-hf/llama3-llava-next-8b-hf"
IMAGE_FOLDER = Path("images")
OUTPUT_FOLDER = IMAGE_FOLDER / "output_images"
INPUT_FOLDER = IMAGE_FOLDER / "input_images"


def load_model_and_processor():
    processor = LlavaNextProcessor.from_pretrained(MODEL_NAME)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    return processor, model


def caption_image(
    image: Image.Image,
    processor: LlavaNextProcessor,
    model: LlavaNextForConditionalGeneration,
    trigger_word: str,
) -> str:
    query = f"""Describe this image in detail. Focus on the main person and their appearance.
    Then, create a new description where you replace the main person with {trigger_word}.
    Maintain the same level of detail about their appearance and surroundings, but use {trigger_word} as if it were a person.
    Format your response as:
    Original description: [Your detailed description]
    Modified description: [Your modified description with {trigger_word}]
    """

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(prompt, image, return_tensors="pt").to(model.device)
    output: GenerationMixin = model.generate(**inputs, max_new_tokens=300)
    out = processor.decode(output[0])

    if "<|end_header_id|>" not in out:
        return ""
    return out.split("<|end_header_id|>")[2].removesuffix("<|eot_id|>")


def process_images(
    processor: LlavaNextProcessor, model: LlavaNextForConditionalGeneration
):
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Clear output folder
    for file in OUTPUT_FOLDER.glob("*"):
        file.unlink()

    for image_path in INPUT_FOLDER.glob("*.png"):
        try:
            with Image.open(image_path) as image:
                caption = caption_image(
                    image,
                    processor,
                    model,
                    TRIGGER_WORD,
                )

                image_name = f"image_{uuid4()}"
                image.save(OUTPUT_FOLDER / f"{image_name}.png")

                with open(OUTPUT_FOLDER / f"{image_name}.txt", "w") as f:
                    caption = caption.split("Modified description:")[1]
                    f.write(caption)

                print(
                    f"Image {image_path} captioned and saved to {OUTPUT_FOLDER / image_name}"
                )
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")


def main():
    processor, model = load_model_and_processor()
    process_images(processor, model)


if __name__ == "__main__":
    main()
