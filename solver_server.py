# solver_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from queue import Queue
import threading
import time
import traceback
import uuid

from custom_peak_solver import CustomPeakSolver

app = FastAPI()
solver = CustomPeakSolver()

# FIFO job queue
job_queue = Queue()
results = {}

class SolveRequest(BaseModel):
    qasm: str

class SolveResponse(BaseModel):
    job_id: str

class ResultResponse(BaseModel):
    job_id: str
    status: str
    bitstring: str | None = None
    elapsed: float | None = None
    error: str | None = None


# --- Background worker thread ---
def worker():
    while True:
        job_id, qasm = job_queue.get()
        t0 = time.perf_counter()
        try:
            bitstring = solver.solve(qasm)
            elapsed = time.perf_counter() - t0
            results[job_id] = {
                "status": "done",
                "bitstring": bitstring,
                "elapsed": elapsed,
                "error": None,
            }
            print(f"[SERVER] Job {job_id} finished in {elapsed:.2f}s")
        except Exception as e:
            traceback.print_exc()
            results[job_id] = {
                "status": "error",
                "bitstring": None,
                "elapsed": None,
                "error": str(e),
            }
        finally:
            time.sleep(5)
            job_queue.task_done()

threading.Thread(target=worker, daemon=True).start()


@app.post("/submit", response_model=SolveResponse)
def submit(req: SolveRequest):
    job_id = str(uuid.uuid4())
    results[job_id] = {"status": "queued"}
    job_queue.put((job_id, req.qasm))
    print(f"[SERVER] Received job {job_id}, queued.")
    return SolveResponse(job_id=job_id)


@app.get("/result/{job_id}", response_model=ResultResponse)
def get_result(job_id: str):
    if job_id not in results:
        return ResultResponse(job_id=job_id, status="not_found")
    entry = results[job_id]
    return ResultResponse(job_id=job_id, **entry)

@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("solver_server:app", host="0.0.0.0", port=8000, reload=False)