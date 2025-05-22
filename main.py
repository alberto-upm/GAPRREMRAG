"""
Script principal para:
  1) Cargar un CSV con preguntas y (opcionalmente) contexto.
  2) Construir el índice RAG.
  3) Generar respuestas para dos columnas de preguntas:
       - 'input' → 'actual_output'
       - 'input_reformulado_2' → 'actual_output_reformulado'
  4) Guardar un nuevo CSV con las columnas de salida añadidas.
"""
import argparse
import pandas as pd
from tqdm.auto import tqdm
from modulo_3_RAG_ollama import load_documents_from_dir, RAG  # load_documents_from_csv

def main():
    parser = argparse.ArgumentParser(
        description="Generar respuestas con RAG desde un CSV."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="/home/jovyan/DEEPEVAL_AL/output/2_dataset_reformulado.csv",
        help="Ruta al archivo CSV que contiene las preguntas (y contexto)."
    )
    parser.add_argument(
        "--documents_dir",
        type=str,
        default="/home/jovyan/Documentos/Docs_pdf",
        help="Ruta a la carpeta con los PDFs a indexar."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/jovyan/DEEPEVAL_AL/output/3_dataset_reformulado_RAG.csv",
        help="Ruta donde se guardará el CSV resultante."
    )
    args = parser.parse_args()
    
    # 1) Leemos el CSV
    df = pd.read_csv(args.csv_path)
    
    # Validar que existan las columnas esperadas
    required = ["input", "input_reformulado_2"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Falta la columna requerida en el CSV: '{col}'")
    
    # 2) Cargamos los documentos desde la carpeta
    print(f"📂 Cargando documentos desde: {args.documents_dir}")
    docs = load_documents_from_dir(args.documents_dir)
    print(f"✅ {len(docs)} documentos cargados.")
    
    # 3) Inicializamos el RAG con los documentos
    rag = RAG(
        docs, 
        #embedding_model="jinaai/jina-embeddings-v3", 
        #model_name="meta-llama/Llama-3.1-8B"
    )
    
    # 4) Preparamos las columnas de salida si no existen
    if 'actual_output' not in df.columns:
        df['actual_output'] = ''
    if 'actual_output_reformulado' not in df.columns:
        df['actual_output_reformulado'] = ''
    
    # Procesamos en batches para mejor manejo de errores
    def process_safely(question):
        try:
            if pd.isna(question):
                return ""
            return rag.answer(str(question))
        except Exception as e:
            print(f"Error procesando pregunta: {e}")
            return f"ERROR: {str(e)}"
    
    # Aplicamos con barras de progreso
    tqdm.pandas(desc="Procesando 'input'")
    df["actual_output"] = df["input"].progress_apply(process_safely)
    
    tqdm.pandas(desc="Procesando 'input_reformulado_2'")
    df["actual_output_reformulado"] = df["input_reformulado_2"].progress_apply(process_safely)
    
    # 5) Guardamos el CSV con las nuevas columnas
    df.to_csv(args.output_path, index=False)
    print(f"✅ Respuestas generadas y guardadas en: {args.output_path}")

if __name__ == "__main__":
    main()

'''
        prompt_template ="""
            <|system|>
            Utilizando la información contenida en el contexto, proporciona una respuesta exhaustiva a la pregunta.
            Responde únicamente a la pregunta formulada; la respuesta debe ser concisa y relevante.
            Indica el número del documento fuente cuando sea pertinente.
            Si la respuesta no puede deducirse del contexto, no des ninguna respuesta.
            Responde a la pregunta basándote en tu conocimiento. Usa el siguiente contexto para ayudar:
            {context}
            
            </s>
            <|user|>
            {question}
            </s>
            <|assistant|>
            Respuesta:
        """
'''
