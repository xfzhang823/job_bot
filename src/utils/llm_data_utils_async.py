import ollama
import asyncio
import logging

logger = logging.getLogger(__name__)


async def call_llama3_async(
    prompt: str,
    expected_res_type: str = "str",
    temperature: float = 0.4,
    max_tokens: int = 1056,
):
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            ollama.generate,
            model="llama3",
            prompt=prompt,
            options={"temperature": temperature, "max_tokens": max_tokens},
        )
        # Process response...
    except asyncio.TimeoutError:
        logger.error("LLaMA 3 call timed out.")
        raise
    except Exception as e:
        logger.error(f"LLaMA 3 call failed: {e}")
        raise


# Usage:
async def main():
    response = await call_llama3_async("Your prompt here")
    print(response)


asyncio.run(main())
