# modulo_3_RAG.py

"""
Módulo RAG: implementación de un sistema de Recuperación Aumentada de Generación (RAG)
usando Ollama Serve para embeddings y Llama 3.1:8b como LLM local.
"""

import pandas as pd
from pathlib import Path

# Cargador de PDFs
from langchain_community.document_loaders import PyPDFLoader

# Esquema de documentos de LangChain
from langchain.schema import Document

# Herramienta para dividir textos en chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store FAISS
from langchain_community.vectorstores import FAISS

# Integración de Ollama para embeddings y LLM
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

# Plantillas de prompt y parsers de salida
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def load_documents_from_csv(csv_path: str, text_column: str) -> list[Document]:
    """
    Carga documentos desde un archivo CSV y los convierte en objetos Document de LangChain.

    Args:
        csv_path: Ruta al archivo CSV.
        text_column: Nombre de la columna que contiene el texto a indexar.

    Returns:
        Lista de langchain.schema.Document.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_column])
    # Cada fila se convierte en Document
    docs = [Document(page_content=str(text)) for text in df[text_column].tolist()]
    return docs


def load_documents_from_dir(dir_path: str) -> list[Document]:
    """
    Recorre recursivamente dir_path buscando todos los PDFs y devuelve
    una lista de Document con su contenido.
    """
    docs: list[Document] = []
    pdf_files = Path(dir_path).rglob("*.pdf")
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            loaded = loader.load()
            docs.extend(loaded)
        except Exception as e:
            print(f"⚠️ Error cargando {pdf_path}: {e}")
    return docs


class RAG:
    """
    Clase que encapsula el flujo RAG usando:
      1) Chunking de documentos.
      2) Indexación con FAISS + OllamaEmbeddings.
      3) LLM local Llama3.1:8b servido por OllamaLLM.
      4) Cadena RAG con PromptTemplate en Español.
    """

    def __init__(
        self,
        docs: list[Document],
        embedding_model: str = "jina/jina-embeddings-v2-base-es:latest",    
        llm_model: str = "llama3.1:8b",
        chunk_size: int = 512,
        chunk_overlap: int = 30,
        k: int = 4
    ):
        # 1) Split de documentos en chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunked_docs = splitter.split_documents(docs)

        # 2) Embeddings + FAISS index
        # Usamos OllamaEmbeddings conectado al servidor local de Ollama :contentReference[oaicite:2]{index=2}
        embeddings = OllamaEmbeddings(
            model=embedding_model
        )
        self.db = FAISS.from_documents(chunked_docs, embeddings)
        self.retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

        # 3) LLM local con OllamaServe
        # Creamos un runnable LLM que interactúa con llama3.1:8b :contentReference[oaicite:3]{index=3}
        llm = OllamaLLM(
            model=llm_model
        )

        # 4) Prompt + cadena RAG
        prompt_template = """
            Utilizando la información contenida en el contexto, proporciona una respuesta exhaustiva a la pregunta.
            Responde UNICAMENTE a la pregunta formulada; la respuesta debe ser concisa y relevante.
            Responde en Español.
            Indica el número del documento fuente cuando sea pertinente.
            Si la respuesta no puede deducirse del contexto, entonces no des ninguna respuesta.
            Usa el siguiente contexto para ayudar a tu conocimiento:
            Contexto:{context}
            Pregunta:{question}

            Importante: responde UNICAMENTE con la respuesta sin añadir comentarios.
            
            Respuesta:
        """
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )

        # Cadena que conecta prompt → LLM → parser de salida
        self.llm_chain = prompt | llm | StrOutputParser()

        # Cadena RAG completa: primero recuperar docs, luego LLM
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.llm_chain
        )

    
    def answer(self, question: str) -> str:
        """
        1) Recupera los docs relevantes.
        2) Extrae y concatena su texto.
        3) Pasa contexto y pregunta a la cadena RAG.
        """
        
        #docs = self.retriever.get_relevant_documents(question)
        #context = "\n\n".join(doc.page_content for doc in docs)
        #return self.rag_chain.invoke({"context": context, "question": question})

        respuesta = self.rag_chain.invoke(question)
        print(respuesta)
        
        return respuesta
    '''

    def answer(self, question: str | dict) -> str:
        """
        Responde preguntas usando el sistema RAG:
          1) Recupera los documentos relevantes.
          2) Formatea el contexto con índices de documento.
          3) Invoca la cadena LLM.
        """
        # Asegurar que la pregunta sea string
        if isinstance(question, dict):
            question = question.get("question", str(question))
        question = str(question)

        try:
            # Recuperar docs
            docs = self.retriever.invoke(question)
            # Formatear contexto
            context = "\n\n".join(
                f"[Doc {i+1}] {doc.page_content}" 
                for i, doc in enumerate(docs)
            )
            # Invocar cadena RAG
            response = self.llm_chain.invoke({
                "context": context,
                "question": question
            })
            return response
        except Exception as e:
            return f"Error al procesar la pregunta: {e}"
    '''
