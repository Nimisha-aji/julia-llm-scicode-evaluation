"""
experiment_runner.py - Full Automated Experiment Pipeline
Julia LLM Scientific Computing Evaluation - Using Groq API

SETUP:
    pip install groq numpy pandas tqdm
    set GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx

USAGE:
    python experiment_runner.py

Safe to interrupt and restart - saves progress after every API call.
"""

import os, re, csv, json, time, math, logging, subprocess, tempfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from groq import Groq

GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
RESULTS_DIR   = Path("results")
CODE_DIR      = RESULTS_DIR / "generated_code"
RESULTS_CSV   = RESULTS_DIR / "experiment_results.csv"
PROGRESS_FILE = RESULTS_DIR / "progress.json"
LOG_FILE      = RESULTS_DIR / "run_log.txt"
RESULTS_DIR.mkdir(exist_ok=True)
CODE_DIR.mkdir(exist_ok=True)

N_SAMPLES    = 5
EXEC_TIMEOUT = 30
API_SLEEP    = 2.0
MAX_TOKENS   = 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# 7 genuinely different model families - meaningful comparison
# Meta, Alibaba, DeepSeek, Mistral, Google - 4 organisations, 3 architectures
MODELS = {
    "llama_3_3_70b": "llama-3.3-70b-versatile",
    "llama_4_scout":  "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen3_32b":      "qwen/qwen3-32b",
    "deepseek_r1":    "deepseek-r1-distill-qwen-32b",
    "mixtral_8x7b":   "mixtral-8x7b-32768",
    "llama_3_1_8b":   "llama-3.1-8b-instant",
    "gemma2_9b":      "gemma2-9b-it",
}
MODEL_ORDER = list(MODELS.keys())

TASKS = [
    {"id":"cg_solver","domain":"numerical_linear_algebra",
     "prompt":"Write a Julia function `cg_solver(A, b, x0, tol)` that solves A*x=b using Conjugate Gradient. Return only executable Julia code, no explanation.",
     "reference_fn":"cg_solver","reference_input":"([4.0 1.0; 1.0 3.0],[1.0,2.0],[0.0,0.0],1e-8)",
     "reference_output":[0.09090909,0.63636364]},
    {"id":"gauss_seidel","domain":"numerical_linear_algebra",
     "prompt":"Write a Julia function `gauss_seidel(A, b, x0, tol, max_iter)` that solves Ax=b using Gauss-Seidel iteration. Return only executable Julia code, no explanation.",
     "reference_fn":"gauss_seidel","reference_input":"([4.0 -1.0; -1.0 3.0],[15.0,10.0],[0.0,0.0],1e-8,1000)",
     "reference_output":[4.09090909,4.69696970]},
    {"id":"gram_schmidt","domain":"numerical_linear_algebra",
     "prompt":"Write a Julia function `gram_schmidt(A)` that orthogonalises columns of A using classical Gram-Schmidt. Return matrix Q with orthonormal columns. Return only executable Julia code, no explanation.",
     "reference_fn":"gram_schmidt","reference_input":"([1.0 1.0; 1.0 0.0; 0.0 1.0],)",
     "reference_output":None},
    {"id":"householder_qr","domain":"numerical_linear_algebra",
     "prompt":"Write a Julia function `householder_qr(A)` that computes QR decomposition using Householder reflections. Return tuple (Q,R). Return only executable Julia code, no explanation.",
     "reference_fn":"householder_qr","reference_input":"([1.0 2.0; 3.0 4.0; 5.0 6.0],)",
     "reference_output":None},
    {"id":"weighted_jacobi","domain":"numerical_linear_algebra",
     "prompt":"Write a Julia function `weighted_jacobi(A, b, x0, omega, tol, max_iter)` solving Ax=b using weighted Jacobi with relaxation omega. Return solution vector. Return only executable Julia code, no explanation.",
     "reference_fn":"weighted_jacobi","reference_input":"([4.0 -1.0; -1.0 3.0],[15.0,10.0],[0.0,0.0],0.5,1e-8,1000)",
     "reference_output":[4.09090909,4.69696970]},
    {"id":"lanczos","domain":"numerical_linear_algebra",
     "prompt":"Write a Julia function `lanczos(A, v0, k)` computing k steps of Lanczos algorithm for symmetric matrix A. Return tuple (alpha, beta). Return only executable Julia code, no explanation.",
     "reference_fn":"lanczos","reference_input":"([4.0 1.0; 1.0 3.0],[1.0,0.0],2)",
     "reference_output":None},
    {"id":"incomplete_cholesky","domain":"numerical_linear_algebra",
     "prompt":"Write a Julia function `incomplete_cholesky(A)` computing incomplete Cholesky factorization. Return lower triangular L. Return only executable Julia code, no explanation.",
     "reference_fn":"incomplete_cholesky","reference_input":"([4.0 2.0; 2.0 3.0],)",
     "reference_output":None},
    {"id":"ica","domain":"numerical_linear_algebra",
     "prompt":"Write a Julia function `ica_demix(X, n_components, max_iter, tol)` performing FastICA on matrix X. Use Random.seed!(42). Return demixing matrix W. Return only executable Julia code, no explanation.",
     "reference_fn":"ica_demix","reference_input":"([1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0],2,100,1e-6)",
     "reference_output":None},
    {"id":"burgers_equation","domain":"computational_mechanics",
     "prompt":"Write a Julia function `burgers_solver(nu, nx, nt, dx, dt)` solving 1D Burgers equation with upwind finite differences. Return solution vector u. Return only executable Julia code, no explanation.",
     "reference_fn":"burgers_solver","reference_input":"(0.1,10,20,0.1,0.001)",
     "reference_output":None},
    {"id":"splitting_operator","domain":"computational_mechanics",
     "prompt":"Write a Julia function `strang_splitting(u0, dt, nt, L1, L2)` applying Strang splitting with operators L1 and L2 for nt steps. Return final state vector. Return only executable Julia code, no explanation.",
     "reference_fn":"strang_splitting","reference_input":"([1.0,2.0,3.0],0.01,10,u->u.*0.99,u->u.*0.98)",
     "reference_output":None},
    {"id":"supg_stabilization","domain":"computational_mechanics",
     "prompt":"Write a Julia function `supg_tau(u, kappa, h)` computing SUPG stabilization parameter tau for advection-diffusion. Return Float64 scalar. Return only executable Julia code, no explanation.",
     "reference_fn":"supg_tau","reference_input":"(2.0,0.1,0.5)",
     "reference_output":None},
    {"id":"nurbs_basis","domain":"computational_mechanics",
     "prompt":"Write a Julia function `nurbs_basis(i, p, t, knots)` evaluating i-th B-spline basis function of degree p at t using Cox-de Boor recursion. Return Float64. Return only executable Julia code, no explanation.",
     "reference_fn":"nurbs_basis","reference_input":"(2,2,0.5,[0.0,0.0,0.0,1.0,1.0,1.0])",
     "reference_output":[0.25]},
    {"id":"chaotic_pendulum","domain":"computational_mechanics",
     "prompt":"Write a Julia function `pendulum_rk4(theta0, omega0, dt, nt, g, L, b)` simulating damped nonlinear pendulum using RK4. Return vector of angles length nt+1. Return only executable Julia code, no explanation.",
     "reference_fn":"pendulum_rk4","reference_input":"(0.5,0.0,0.01,10,9.81,1.0,0.1)",
     "reference_output":None},
    {"id":"option_pricing","domain":"computational_finance",
     "prompt":"Write a Julia function `black_scholes_mc(S0, K, r, sigma, T, n_paths, n_steps)` estimating European call price using Monte Carlo. Use Random.seed!(42). Return Float64 price. Return only executable Julia code, no explanation.",
     "reference_fn":"black_scholes_mc","reference_input":"(100.0,100.0,0.05,0.2,1.0,1000,50)",
     "reference_output":[10.45]},
]

REFERENCE_CODE = {
    "cg_solver": "function cg_solver(A,b,x0,tol)\n    x=copy(x0);r=b-A*x;p=copy(r);rsold=dot(r,r)\n    for i in 1:length(b)*10\n        Ap=A*p;alpha=rsold/dot(p,Ap);x.+=alpha.*p;r.-=alpha.*Ap\n        rsnew=dot(r,r);sqrt(rsnew)<tol&&break;p=r.+(rsnew/rsold).*p;rsold=rsnew\n    end;return x\nend",
    "gauss_seidel": "function gauss_seidel(A,b,x0,tol,max_iter)\n    x=copy(x0);n=length(b)\n    for iter in 1:max_iter\n        x_old=copy(x)\n        for i in 1:n\n            s=b[i]-sum(A[i,j]*x[j] for j in 1:n if j!=i);x[i]=s/A[i,i]\n        end\n        norm(x-x_old)<tol&&break\n    end;return x\nend",
    "nurbs_basis": "function nurbs_basis(i,p,t,knots)\n    p==0&&return (knots[i]<=t<knots[i+1]) ? 1.0 : 0.0\n    d1=knots[i+p]-knots[i];d2=knots[i+p+1]-knots[i+1]\n    c1=d1>0 ? (t-knots[i])/d1*nurbs_basis(i,p-1,t,knots) : 0.0\n    c2=d2>0 ? (knots[i+p+1]-t)/d2*nurbs_basis(i+1,p-1,t,knots) : 0.0\n    return c1+c2\nend",
}

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f: return set(json.load(f))
    return set()

def save_progress(done_set):
    with open(PROGRESS_FILE,"w") as f: json.dump(list(done_set),f)

def progress_key(m,t,s): return f"{m}__{t}__{s}"

def call_groq_api(client, model_id, prompt, max_retries=3):
    model_name = MODELS[model_id]
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role":"system","content":"You are an expert Julia programmer. Return only executable Julia code inside a ```julia code block. No explanations."},
                    {"role":"user","content":prompt}
                ],
                max_tokens=MAX_TOKENS, temperature=0.2, top_p=0.95,
            )
            return response.choices[0].message.content
        except Exception as e:
            err=str(e)
            wait=30*(attempt+1) if "rate" in err.lower() or "429" in err else 10
            log.warning(f"API error attempt {attempt+1}: {err[:150]} - waiting {wait}s")
            time.sleep(wait)
    return None

def extract_julia_code(text):
    if not text: return None
    m = re.search(r"```julia\s*(.*?)```", text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if "function" in code or "return" in code: return code
    if "function " in text[:200]: return text.strip()
    return None

JULIA_HARNESS = """
using LinearAlgebra, Statistics, Random
{code}
try
    result = {fn_name}({inputs})
    if isa(result, AbstractArray)
        println("RESULT_ARRAY:" * join(string.(vec(Float64.(result))), ","))
    elseif isa(result, Number)
        println("RESULT_SCALAR:" * string(Float64(result)))
    elseif isa(result, Tuple)
        println("RESULT_TUPLE:ok")
    else
        println("RESULT_OTHER:ok")
    end
    println("STATUS:pass")
catch e
    println("STATUS:fail")
    println("ERROR:" * string(e))
end
"""

JULIA_TIMING = """
using LinearAlgebra, Statistics, Random, BenchmarkTools
{code}
try
    b = @benchmark {fn_name}({inputs}) samples=3 evals=1
    println("BENCH_MIN_MS:" * string(minimum(b).time / 1e6))
    println("BENCH_MEM_MB:" * string(b.memory / 1e6))
catch e
    println("BENCH_STATUS:fail")
end
"""

def run_julia(code, fn_name, inputs, timeout=EXEC_TIMEOUT):
    result = {"status":"fail","output":None,"exec_time_ms":float("nan"),"memory_mb":float("nan"),"error":""}
    harness = JULIA_HARNESS.format(code=code, fn_name=fn_name, inputs=inputs)
    with tempfile.NamedTemporaryFile(mode="w",suffix=".jl",delete=False,encoding="utf-8") as f:
        f.write(harness); tmp=f.name
    try:
        proc = subprocess.run(["julia","--startup-file=no",tmp],capture_output=True,text=True,timeout=timeout)
        out  = proc.stdout
        if "STATUS:pass" in out:
            result["status"]="pass"
            m=re.search(r"RESULT_ARRAY:([\d.,\-eE]+)",out)
            if m:
                try: result["output"]=[float(x) for x in m.group(1).split(",")]
                except: pass
            m=re.search(r"RESULT_SCALAR:([\d.\-eE]+)",out)
            if m:
                try: result["output"]=[float(m.group(1))]
                except: pass
        else:
            m=re.search(r"ERROR:(.*)",out)
            result["error"]=m.group(1).strip() if m else proc.stderr[:200]
    except subprocess.TimeoutExpired: result["error"]="timeout"
    except FileNotFoundError: result["error"]="julia_not_found"
    finally:
        try: os.unlink(tmp)
        except: pass

    if result["status"]=="pass":
        timing = JULIA_TIMING.format(code=code, fn_name=fn_name, inputs=inputs)
        with tempfile.NamedTemporaryFile(mode="w",suffix=".jl",delete=False,encoding="utf-8") as f:
            f.write(timing); tmp2=f.name
        try:
            proc2=subprocess.run(["julia","--startup-file=no",tmp2],capture_output=True,text=True,timeout=120)
            out2=proc2.stdout
            m=re.search(r"BENCH_MIN_MS:([\d.e\-E]+)",out2)
            if m: result["exec_time_ms"]=float(m.group(1))
            m=re.search(r"BENCH_MEM_MB:([\d.e\-E]+)",out2)
            if m: result["memory_mb"]=float(m.group(1))
        except: pass
        finally:
            try: os.unlink(tmp2)
            except: pass
    return result

def compute_accuracy(model_output, reference_output):
    if reference_output is None or model_output is None: return None, None
    try:
        y_m=np.array(model_output,dtype=float); y_r=np.array(reference_output,dtype=float)
        n=min(len(y_m),len(y_r)); y_m=y_m[:n]; y_r=y_r[:n]
        rel=np.linalg.norm(y_m-y_r)/(np.linalg.norm(y_r)+1e-9)
        return round(max(0.0,1.0-min(rel,1.0)),6), round(rel,6)
    except: return 0.0, 1.0

def compute_codebleu_score(hyp, ref):
    if not hyp or not ref: return 0.0
    def tok(c): return re.findall(r"[a-zA-Z_]\w*|[\+\-\*/=<>!]+|[\d.]+|[\(\)\[\]{},;]",c)
    def ng(t,n): return [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
    def prec(ht,rt,n):
        hg=ng(ht,n); rg=set(ng(rt,n))
        return sum(1 for g in hg if g in rg)/len(hg) if hg else 0.0
    ht=tok(hyp); rt=tok(ref)
    bp=min(1.0,len(ht)/(len(rt)+1e-9))
    return round(bp*(0.5*prec(ht,rt,1)+0.5*prec(ht,rt,2)),6)

def pass_at_k(n,c,k):
    if n-c<k: return 1.0
    return 1.0 - math.comb(n-c,k)/math.comb(n,k)

def consistency_score(acc_scores):
    arr=np.array([a for a in acc_scores if a is not None],dtype=float)
    if len(arr)==0: return 0.0
    return round(1.0-arr.std()/(arr.mean()+1e-9),6)

CSV_FIELDS=["model","task","domain","sample_index","correctness","accuracy_score",
            "relative_l2_error","exec_time_ms","memory_mb","codebleu",
            "consistency","pass_at_1","pass_at_5","error_message"]

def init_csv():
    if not RESULTS_CSV.exists():
        with open(RESULTS_CSV,"w",newline="",encoding="utf-8") as f:
            csv.DictWriter(f,fieldnames=CSV_FIELDS).writeheader()

def append_row(row):
    with open(RESULTS_CSV,"a",newline="",encoding="utf-8") as f:
        csv.DictWriter(f,fieldnames=CSV_FIELDS).writerow({k:row.get(k,"") for k in CSV_FIELDS})

def run_experiment():
    if not GROQ_API_KEY:
        print("\nERROR: GROQ_API_KEY not set.\nRun: set GROQ_API_KEY=gsk_your_key_here\n"); return

    client=Groq(api_key=GROQ_API_KEY)
    init_csv()
    done=load_progress()
    total=len(MODELS)*len(TASKS)*N_SAMPLES
    log.info(f"Total API calls: {total} | Already done: {len(done)}")

    existing_df=pd.read_csv(RESULTS_CSV) if RESULTS_CSV.exists() and RESULTS_CSV.stat().st_size>0 else pd.DataFrame()
    existing_rows=set()
    if not existing_df.empty:
        for _,r in existing_df.iterrows():
            existing_rows.add((r["model"],r["task"],int(r["sample_index"])))

    with tqdm(total=total,desc="Experiment",unit="sample") as pbar:
        for model_id in MODELS:
            for task in TASKS:
                task_id=task["id"]; domain=task["domain"]
                fn_name=task["reference_fn"]; inputs=task["reference_input"]
                ref_out=task["reference_output"]; ref_code=REFERENCE_CODE.get(task_id,"")
                sample_results=[]

                for s in range(N_SAMPLES):
                    pbar.update(1)
                    key=progress_key(model_id,task_id,s)

                    if (model_id,task_id,s) in existing_rows:
                        row=existing_df[(existing_df["model"]==model_id)&(existing_df["task"]==task_id)&(existing_df["sample_index"]==s)]
                        if not row.empty:
                            r=row.iloc[0]
                            acc=float(r["accuracy_score"]) if pd.notna(r.get("accuracy_score")) else None
                            sample_results.append({"correctness":float(r["correctness"]),"accuracy_score":acc})
                        done.add(key); continue

                    log.info(f"  {model_id} | {task_id} | s{s+1}")
                    raw=call_groq_api(client,model_id,task["prompt"])
                    code=extract_julia_code(raw) if raw else None

                    code_path=CODE_DIR/f"{model_id}__{task_id}__{s}.jl"
                    if code: code_path.write_text(code,encoding="utf-8")

                    row_data={"model":model_id,"task":task_id,"domain":domain,"sample_index":s,
                              "correctness":0.0,"accuracy_score":None,"relative_l2_error":None,
                              "exec_time_ms":float("nan"),"memory_mb":float("nan"),"codebleu":0.0,
                              "consistency":None,"pass_at_1":None,"pass_at_5":None,
                              "error_message":"no_code" if not code else ""}

                    if code:
                        er=run_julia(code,fn_name,inputs)
                        row_data["correctness"]=1.0 if er["status"]=="pass" else 0.0
                        row_data["exec_time_ms"]=er["exec_time_ms"]
                        row_data["memory_mb"]=er["memory_mb"]
                        row_data["error_message"]=er["error"]
                        if er["status"]=="pass":
                            acc,rel=compute_accuracy(er["output"],ref_out)
                            row_data["accuracy_score"]=acc
                            row_data["relative_l2_error"]=rel
                            row_data["codebleu"]=compute_codebleu_score(code,ref_code)

                    sample_results.append({"correctness":row_data["correctness"],
                                           "accuracy_score":row_data["accuracy_score"],
                                           "_row":row_data,"_s":s})
                    done.add(key); save_progress(done)
                    time.sleep(API_SLEEP)

                n_c=int(sum(r["correctness"] for r in sample_results))
                p1=pass_at_k(N_SAMPLES,n_c,1); p5=pass_at_k(N_SAMPLES,n_c,5)
                cons=consistency_score([r["accuracy_score"] for r in sample_results])
                for r in sample_results:
                    if "_row" not in r: continue
                    r["_row"]["consistency"]=cons
                    r["_row"]["pass_at_1"]=round(p1,6)
                    r["_row"]["pass_at_5"]=round(p5,6)
                    append_row(r["_row"])

    log.info("=== DONE === Run: python analyze_results.py")
    print("\nDone! Now run: python analyze_results.py")

if __name__=="__main__":
    run_experiment()
