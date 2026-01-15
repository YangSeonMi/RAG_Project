import csv
import io
import os
from fastapi import FastAPI, Body, HTTPException, UploadFile, File
from starlette.responses import Response, RedirectResponse, JSONResponse
from starlette.requests import Request
from pydantic import BaseModel
import requests

app = FastAPI()

RAG_SEVER_URL = os.getenv("RAG_SEVER_URL", "http://rag_server:7777")
OLLAMA_SEVER_URL = os.getenv("OLLAMA_SEVER_URL", "http://ollama:11434")

users = [{"username": "admin", "password": "0115"}]

class LoginUsers(BaseModel):
    username: str
    password: str

class QueryRequest(BaseModel):
    query: str


@app.post("/login")
def login(response: Response, user: LoginUsers = Body()): #option enter  response ->sh,  body ->
    # 로그인 검증
    ok = any(
        u["username"] == user.username and u["password"] == user.password
        for u in users
    )
    if not ok:
        return JSONResponse({"ok": False, "reason": "invalid credentials"}, status_code=401)

    # 응답 만들고 쿠키 세팅
    res = JSONResponse({"ok": True})
    res.set_cookie("username", user.username, httponly=True)
    return res


@app.get("/page")
def page(request: Request):
    username = request.cookies.get("username")
    if not username:
        return JSONResponse({"ok": False, "message": "NO_C"}, status_code=401)

    if username in [u.username for u in users]:
        return JSONResponse({"ok": True, "message": "OK_C"})

    return JSONResponse({"ok": False, "message": "unknown user"}, status_code=403)

def get_current_user(request: Request):
    username = (
        request.headers.get("X-User")
        or request.cookies.get("username")
    )

    if not username:
        raise HTTPException(status_code=401, detail="Not logged in")

    if username not in [u["username"] for u in users]:
        raise HTTPException(status_code=401, detail="Invalid user")

    return username


@app.post("/upload")
async def upload_csv_to_rag(
    request: Request,
    file: UploadFile = File(...),
    chunk_size: int = 1024
):
    username = get_current_user(request)

    contents = await file.read()
    decoded = contents.decode("utf-8")

    stream = io.StringIO(decoded)
    csv_reader = csv.reader(stream)

    # CSV → Text
    full_text = csv_to_text(csv_reader)

    # RAG 서버로 전달 (split은 RAG가 담당)
    response = requests.post(
        f"{RAG_SEVER_URL}/upload",
        json={
            "full_text": full_text,
            "chunk_size": chunk_size
        },
        timeout=60
    )

    response.raise_for_status()

    return {
        "ok": True,
        "filename": file.filename,
        "rag_response": response.json()
    }


def csv_to_text(csv_reader) -> str:
    lines = []
    for row in csv_reader:
        line = " | ".join(col.strip() for col in row)
        lines.append(line)
    return "\n".join(lines)


def upload_to_rag(full_text, chunk_size: int = 1024):
    # rag 서버에 연결
    response = requests.post(
        f"{RAG_SEVER_URL}/upload",
        json={"full_text": full_text, "chunk_size": chunk_size},\
        timeout=60,
    )
    return response.json()

def llm_response(query):
    response = requests.post(
        url=f"{OLLAMA_SEVER_URL}/answer",
        json={"query": query},
        timeout=180
    )
    response.raise_for_status()
    return response.json()

@app.post("/query")
def query(
    request: Request,
    body: QueryRequest
):
    username = get_current_user(request)

    response = requests.post(
        f"{RAG_SEVER_URL}/answer",
        json={"query": body.query},
        timeout=180
    )
    response.raise_for_status()

    return {
        "ok": True,
        "user": username,
        "query": body.query,
        "llm_response": response.json()
    }


#다른 기능 추가하기!!!!!



# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
# if __name__ == '__main__':

