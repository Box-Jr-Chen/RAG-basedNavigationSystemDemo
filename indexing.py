# from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import os

# 設定 PDF 目錄和 Chroma 資料庫儲存目錄
PDF_FOLDER = "./PDF"
TXT_FOLDER = "./Txt"
CHROMA_DB_DIR = "./chroma_db"


def load_and_index_documents():
    # # 找到所有 PDF 檔案
    # pdf_files = [os.path.join(PDF_FOLDER, f) for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    # print(len(pdf_files))

    # 找到所有 PDF 檔案
    txt_files = [os.path.join(TXT_FOLDER, f) for f in os.listdir(TXT_FOLDER) if f.endswith(".txt")]
    print(f"📄 發現 {len(txt_files)} 個 txt 檔案")

    docs = []
    for txt_file in txt_files:
        print(f"🔍 讀取：{txt_file}")
        loader = TextLoader(txt_file, encoding="utf-8")
        docs.extend(loader.load())

    # 切割文本
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)

    # 生成向量嵌入
    embeddings = OllamaEmbeddings(model="mistral")  # 替換為你要使用的模型
    vectorstore = Chroma.from_documents(
        # texts, embeddings, collection_name="multi_pdf_docs", persist_directory=CHROMA_DB_DIR
        texts, embeddings, collection_name="multi_txt_docs", persist_directory=CHROMA_DB_DIR
    )

    # 儲存向量資料庫
    vectorstore.persist()
    print(f"✅ 已成功索引 {len(texts)} 段文本，儲存於 {CHROMA_DB_DIR}")

if __name__ == "__main__":
    load_and_index_documents()