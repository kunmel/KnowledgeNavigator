import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import llm_config as config
 
app = FastAPI()

sampling_params = SamplingParams(temperature=config.llama2_temperature, top_p=config.llama2_top_p, max_tokens=config.llama2_max_tokens)

llm = LLM(model=config.llama2_path)
    
def query_llm(prompts):
    outputs = llm.generate(prompts, sampling_params)
    response = []
    for output in outputs:
        response.append(output.outputs[0].text)
    return {"answer": response}
    
class LLMQuery(BaseModel):
    texts:list[str]

@app.post("/query_llm/")
async def get_answer(llm_query:LLMQuery):
    return query_llm(llm_query.texts)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=config.llama2_port)