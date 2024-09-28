import asyncio
from tritonclient.http.aio import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import *


class tritonClient:
    def __init__(self, model_names):
        self.host = "localhost:8000"
        self.model_names = model_names

    async def inference(self, input_ids, attention_mask):
        client = InferenceServerClient(self.host)
        input_shape = input_ids.shape
        inputs = [InferInput('input_ids', input_shape, np_to_triton_dtype(input_ids.dtype)), 
                  InferInput('attention_mask', input_shape, np_to_triton_dtype(attention_mask.dtype))]

        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)

        outputs = [InferRequestedOutput('output')]
        results = await asyncio.gather(*[client.infer(model_name, inputs, outputs=outputs) for model_name in self.model_names])
        
        for i, result in enumerate(results):
            if i == 0:
                result = result.as_numpy('output')
                output_data = result.copy()
            else:
                result = result.as_numpy('output')
                output_data += result.copy()
                
        output_data /= len(self.model_names)
        
        await client.close()
        return output_data