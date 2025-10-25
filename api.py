from fastapi import FastAPI
from agent import Agent
from fastapi.responses import JSONResponse
from fastapi.requests import Request

app = FastAPI()
agent = Agent()

@app.get("/chatbot/approval")
def approval(approval: str):
    try:
        print(approval)
        if approval == 'true':
            data = approval
            data = agent.approvalTrue() 
            return JSONResponse(data)
        
        if approval == 'false':
            data = agent.approvalFalse()
            return JSONResponse(data, 200)
        
    except Exception as e:
        return JSONResponse(str(e), 500)

@app.post("/chatbot/search")
async def search(request: Request):
    """
        Request JSON:
        {
            "query": str
        }
    
    """
    data = await request.json()
    query = data.get("query")
    data = agent.searchQuery(query)
    return JSONResponse(data)
