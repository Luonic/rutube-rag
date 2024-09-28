import httpx
import asyncio


api_url = 'http://localhost:8000/v1'

async def generate_inference_async(user_message):
	headers = {
    	"Content-Type": "application/json"
	}

	data = {
    	"model": "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24",
    	"messages": [
        	{"role": "user", "content": user_message}
    	]
	}

	async with httpx.AsyncClient() as client:
		response = await client.post(f"{api_url}/chat/completions", timeout=600, headers=headers, json=data)
		response.raise_for_status()  # Raise an error for bad responses
		result = response.json()

	return result['choices'][0]['message']['content']

if __name__ == "__main__":
	question = str(input('Ваш вопрос:'))
	print(asyncio.run(generate_inference_async(question)))
