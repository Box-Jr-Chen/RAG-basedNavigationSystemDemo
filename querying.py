from langchain.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langserve import add_routes
from fastapi import FastAPI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda

# 定義多個 PromptTemplate，包含不同的語氣
templates = {
    "professional": PromptTemplate(
        input_variables=["raw_question"],
        template="請用條列式回答以下問題：{raw_question}。答案不要超過 100 字"
    ),
    "friendly": PromptTemplate(
        input_variables=["raw_question"],
        template="請用輕鬆親切的方式回覆這個問題：{raw_question}"
    ),
    "short_answer": PromptTemplate(
        input_variables=["raw_question"],
        template="請用一句話簡要回應下列問題：{raw_question}"
    )
}


def safe_format_prompt(template_name: str, user_input: dict):
    if template_name not in templates:
        raise ValueError(f"❌ 找不到名為 '{template_name}' 的模板")
    
    template = templates[template_name]
    required_keys = set(template.input_variables)
    provided_keys = set(user_input.keys())
    
    missing_keys = required_keys - provided_keys
    extra_keys = provided_keys - required_keys

    if missing_keys:
        raise ValueError(f"❌ 模板 '{template_name}' 缺少必要輸入變數：{', '.join(missing_keys)}")
    if extra_keys:
        raise ValueError(f"❌ 模板 '{template_name}' 提供了未使用的變數：{', '.join(extra_keys)}")

    return template.format(**user_input)


# 主查詢邏輯，支援自然語言問題 + 模板語氣
def query_chain_fn(input):
    try:
        # 從輸入中解析 question 與 template_name
        question = input.get("question")
        template_name = input.get("template_name", "professional")  # 預設使用 'professional' 模板

        if not question:
            raise ValueError("❌ 請提供問題 ('question') 參數")

        # 使用 safe_format_prompt 自動包裝語氣
        prompt_text = safe_format_prompt(template_name, {"raw_question": question})

        # 使用 vector store 進行相似度搜尋
        docs = vectorstore.similarity_search(prompt_text, k=3)
        
        # 回傳 Chain 結果
        return chain.run(input_documents=docs, question=prompt_text)

    except Exception as e:
        return {"error": str(e)}


# 建立 FastAPI 應用
app = FastAPI()

# 初始化嵌入、向量資料庫、LLM、Chain
embeddings = OllamaEmbeddings(model="mistral")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="multi_txt_docs"
)
llm = Ollama(model="qwen:4b")
chain = load_qa_chain(llm, chain_type="stuff")


# 包裝成 Runnable
runnable_chain = RunnableLambda(query_chain_fn)

# 加入 API 路由
add_routes(app, runnable_chain, path="/ask")