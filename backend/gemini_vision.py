from google import genai
from google.genai import types

client = genai.Client(
    vertexai=True,
    project="gen-lang-client-0135398585",
    location="us-central1",
)


async def ask_gemini_vision(image_bytes: bytes) -> str:
    """Uses Gemini Vision to analyze an image of a MOF structure."""
    prompt = (
        "Analyze the provided image. If it shows a MOF material, identify the structure "
        "and its key characteristics, such as pore size, geometry, and composition. "
        "Also, provide a brief summary of its potential applications. If it is not a MOF, "
        "describe what is in the image."
    )
    
    contents = [
        types.Part(
            inline_data=types.Blob(
                mime_type='image/jpeg',
                data=image_bytes
            )
        ),
        types.Part(text=prompt)
    ]
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg',
        ),
        'Caption this image.'
        ]
    )

    print(response.text)
    return response.text