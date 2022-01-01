import uvicorn
from fastapi import FastAPI, File
from typing import List


app = FastAPI()

@app.post("/files/")
async def create_files(files: List[bytes] = File(...)):
    return {"file_sizes": [len(file) for file in files]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1360)