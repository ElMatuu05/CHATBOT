import os
import random
import re
import pandas as pd
from langchain.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pdf2image import convert_from_path
import pytesseract


class UNLPamComputacionChatbot:
    def __init__(self, pdf_folder="data", urls=None):
        all_docs = []

        # =====================
        # 1. Cargar documentos PDF y Excel
        # =====================
        if os.path.exists(pdf_folder):
            for file in os.listdir(pdf_folder):
                file_path = os.path.join(pdf_folder, file)

                if file.endswith(".pdf"):
                    try:
                        loader = PyPDFLoader(file_path)
                        documents = loader.load()
                        all_docs.extend(documents)
                    except Exception:
                        print(f"⚠️ No se pudo leer {file_path}, usando OCR...")
                        all_docs.extend(self._load_pdf_with_ocr(file_path))

                elif file.endswith((".xlsx", ".xls")):
                    try:
                        excel_docs = self._load_excel(file_path)
                        all_docs.extend(excel_docs)
                    except Exception as e:
                        print(f"❌ Error leyendo {file}: {e}")

        # =====================
        # 2. URLs
        # =====================
        if urls:
            try:
                loader = UnstructuredURLLoader(urls=urls)
                url_docs = loader.load()
                all_docs.extend(url_docs)
            except Exception as e:
                print(f"❌ Error cargando URLs: {e}")

        # =====================
        # 3. Procesamiento de texto
        # =====================
        if not all_docs:
            raise ValueError("❌ No se encontraron documentos válidos.")

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(all_docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = FAISS.from_documents(docs, embeddings)

        # =====================
        # 4. Modelo LLM (Groq)
        # =====================
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ Falta la variable de entorno GROQ_API_KEY.")

        self.llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.2,  # más estable
            api_key=api_key
        )

        # =====================
        # 5. Prompt
        # =====================
        template = """
        Sos un asistente informativo del Profesorado en Computación de la UNLPam.
        Contestá en tono claro y directo. Usá la información del contexto y de los documentos cargados.
        Si la información no está en los documentos ni en las FAQs, **no inventes**:
        en ese caso decí que no lo sabés y ofrecé el contacto de Bedelía.

        Contexto:
        {context}

        Pregunta: {question}
        Respuesta:
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        # =====================
        # 6. QA con memoria
        # =====================
        self.qa = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.db.as_retriever(),
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer"),
            return_source_documents=False,
            output_key="answer",
            combine_docs_chain_kwargs={"prompt": prompt}
        )

        self.chat_history = []

        # =====================
        # 7. Plan de estudios + correlativas
        # =====================
        self.plan_dict, self.correlativas = self._definir_plan()

        # =====================
        # 8. FAQs extendidas
        # =====================
        self.faq = {
            "duracion": {
                "keywords": ["duración", "cuantos años", "dura"],
                "answer": "El profesorado dura **4 años**."
            },
            "titulo": {
                "keywords": ["título", "titulo", "certificado"],
                "answer": "El título es **Profesor/a en Computación**, otorgado por la UNLPam."
            },
            "materias": {
                "keywords": ["materias", "plan de estudios", "asignaturas", "cuatrimestres", "correlativas"],
                "answer": self.get_plan_estudios()
            },
            "modalidad": {
                "keywords": ["modalidad", "presencial", "virtual"],
                "answer": "La modalidad es **presencial** en la Facultad de Ciencias Exactas y Naturales de la UNLPam (Santa Rosa, La Pampa)."
            },
            "horarios": {
                "keywords": ["horarios", "cursada", "clases", "turnos"],
                "answer": (
                    "Los horarios de cursada se actualizan cada cuatrimestre. "
                    "👉 Los horarios oficiales actualizados se publican en la web: "
                    "https://exactas.unlpam.edu.ar/academica/bedelia/horarios"
                )
            }
        }

        # =====================
        # 9. Contacto Bedelía
        # =====================
        self.contacto_bedelia = (
            "Podés consultar directamente con **Bedelía de la Facultad de Ciencias Exactas y Naturales**:\n"
            "- 📧 Mail: bedelia.exactas@unlpam.edu.ar\n"
            "- ☎️ Teléfono: (02954) 451000 (interno 1402)\n"
        )

        # =====================
        # 10. Respuestas básicas
        # =====================
        self.responses = {
            "greetings": ["¡Hola! ¿Cómo va todo? 👋", "¡Buenas! ¿Listo/a para sacarte dudas? 😄"],
            "farewells": ["¡Nos vemos! Que andes genial 🙌", "¡Chau! Éxitos en la facu 💪"],
            "thanks": ["¡De nada! Para eso estoy 🤗", "Tranqui, cuando quieras 😉"],
            "other_careers": ["Solo tengo info del **Profesorado en Computación de la UNLPam** 😅"],
        }

    # =====================
    # Definir plan y correlativas
    # =====================
    def _definir_plan(self):
        plan = {
            "Pimer, 1, 1er": {
                "1er cuatrimestre": ["Práctica Educativa I", "Introducción a la Computación"],
                "2do cuatrimestre": ["Psicología", "Pedagogía"],
                "anual": ["Matemática"]
            },
            "2": {
                "1er cuatrimestre": ["Programación I", "Matemática Discreta", "Informática Educativa I"],
                "2do cuatrimestre": ["Práctica Educativa II", "Didáctica", "Programación II", "Probabilidad y Estadística"]
            },
            "Tercer, 3, 3er": {
                "1er cuatrimestre": ["Práctica Educativa III", "Informática Educativa II", "Estructuras de Datos y Algoritmos", "Antropología y Sociología"],
                "2do cuatrimestre": ["Política y Legislación Escolar", "Lenguajes de Programación", "Organización de Computadoras I"]
            },
            "4": {
                "anual": ["Práctica Educativa IV"],
                "1er cuatrimestre": ["Bases de Datos", "Organización de Computadoras II"],
                "2do cuatrimestre": ["Métodos de Investigación Educativa", "Desarrollo de Sistemas", "Optativa"]
            }
        }

        correlativas = {
            "Programación I": ["Introducción a la Computación"],
            "Matemática Discreta": ["Matemática"],
            "Informática Educativa I": ["Práctica Educativa I"],
            "Programación II": ["Programación I", "Matemática Discreta"],
            "Probabilidad y Estadística": ["Matemática"],
            "Práctica Educativa II": ["Práctica Educativa I"],
            "Didáctica": ["Psicología", "Pedagogía"],
            "Estructuras de Datos y Algoritmos": ["Programación II", "Matemática Discreta", "Introducción a la Computación"],
            "Informática Educativa II": ["Informática Educativa I"],
            "Práctica Educativa III": ["Práctica Educativa II"],
            "Antropología y Sociología": ["Pedagogía", "Psicología"],
            "Política y Legislación Escolar": ["Pedagogía"],
            "Lenguajes de Programación": ["Matemática Discreta"],
            "Organización de Computadoras I": ["Programación I", "Matemática", "Introducción a la Computación"],
            "Bases de Datos": ["Estructuras de Datos y Algoritmos"],
            "Organización de Computadoras II": ["Organización de Computadoras I", "Probabilidad y Estadística", "Matemática Discreta"],
            "Métodos de Investigación Educativa": ["Antropología y Sociología", "Informática Educativa I"],
            "Desarrollo de Sistemas": ["Bases de Datos", "Organización de Computadoras I", "Programación II"],
            "Práctica Educativa IV": ["Práctica Educativa III", "Práctica Educativa II", "Didáctica"]
        }

        return plan, correlativas

    # =====================
    # Plan formateado
    # =====================
    def get_plan_estudios(self):
        salida = ["📘 **Plan de Estudios – Profesorado en Computación (UNLPam, Plan 2015)**"]
        for año, bloques in self.plan_dict.items():
            salida.append(f"\n### {año}° Año")
            for cuatri, materias in bloques.items():
                salida.append(f"- **{cuatri.capitalize()}**")
                for m in materias:
                    corr = ""
                    if m in self.correlativas:
                        corr = f" → Correlativas: {', '.join(self.correlativas[m])}"
                    salida.append(f"  - {m}{corr}")
        return "\n".join(salida)

    def get_correlativas(self, materia):
        materia = materia.strip().lower()
        for m, reqs in self.correlativas.items():
            if m.lower() == materia:
                return f"📚 Para cursar **{m}** necesitás tener aprobadas: {', '.join(reqs)}"
        return "⚠️ No encontré correlativas para esa materia."

    # =====================
    # Conversación
    # =====================
    def get_answer(self, query):
        q_lower = query.lower()

        # 1. Pregunta sobre correlativas
        match = re.search(r"correlativa.*de (.+)", q_lower)
        if match:
            materia = match.group(1).strip()
            return self.get_correlativas(materia)

        if "qué necesito para cursar" in q_lower or "requisito" in q_lower:
            materia = q_lower.split("cursar")[-1].strip()
            return self.get_correlativas(materia)

        # 2. Preguntas sobre plan por año/cuatrimestre
        match = re.search(r"(\d)(er|do|ro)?\s*año.*(1er|2do)\s*cuatrimestre", q_lower)
        if match:
            año = match.group(1)
            cuatri = match.group(3)
            materias = self.plan_dict.get(año, {}).get(f"{cuatri} cuatrimestre", [])
            return f"📘 {año}° año – {cuatri} cuatrimestre:\n- " + "\n- ".join(materias)

        match = re.search(r"(\d)(er|do|ro)?\s*año", q_lower)
        if match:
            año = match.group(1)
            bloques = self.plan_dict.get(año, {})
            salida = [f"📘 Plan {año}° año:"]
            for cuatri, materias in bloques.items():
                salida.append(f"\n{cuatri.capitalize()}:\n- " + "\n- ".join(materias))
            return "\n".join(salida)

        # 3. FAQs
        for faq_item in self.faq.values():
            if any(keyword in q_lower for keyword in faq_item["keywords"]):
                if "horarios" in faq_item["keywords"] or "cursada" in faq_item["keywords"]:
                    return faq_item["answer"] + "\n\n⚠️ Confirmá siempre con Bedelía:\n" + self.contacto_bedelia
                return faq_item["answer"]

        # 4. Respuestas básicas
        if any(word in q_lower for word in ["hola", "buenas", "hey", "holis"]):
            return random.choice(self.responses["greetings"])
        elif any(word in q_lower for word in ["chau", "adiós", "nos vemos", "bye"]):
            return random.choice(self.responses["farewells"])
        elif any(word in q_lower for word in ["gracias", "thank"]):
            return random.choice(self.responses["thanks"])
        elif any(word in q_lower for word in ["ingeniería", "medicina", "abogacía", "otra carrera"]):
            return random.choice(self.responses["other_careers"])

        # 5. Intentar con documentos (LLM)
        answer = None
        try:
            result = self.qa({"question": query, "chat_history": self.chat_history})
            if result and result.get("answer") and len(result["answer"].strip()) > 5:
                answer = result["answer"].strip()
        except Exception:
            pass

        # 6. Fallback
        return answer or f"No encontré esa información exacta. Te recomiendo consultar en Bedelía:\n\n{self.contacto_bedelia}"

    # =====================
    # OCR PDFs con imágenes
    # =====================
    def _load_pdf_with_ocr(self, pdf_path):
        docs = []
        try:
            pages = convert_from_path(pdf_path)
            for i, page in enumerate(pages):
                text = pytesseract.image_to_string(page, lang="spa+eng")
                if text.strip():
                    docs.append({"page_content": text, "metadata": {"source": pdf_path, "page": i+1}})
        except Exception as e:
            print(f"❌ Error OCR {pdf_path}: {e}")
        return docs

    # =====================
    # Excel a documentos
    # =====================
    def _load_excel(self, excel_path):
        docs = []
        try:
            df = pd.read_excel(excel_path)
            for i, row in df.iterrows():
                row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
                if row_text.strip():
                    docs.append({"page_content": row_text, "metadata": {"source": excel_path, "row": i+1}})
        except Exception as e:
            print(f"❌ Error procesando Excel {excel_path}: {e}")
        return docs

    # =====================
    # Reset chat
    # =====================
    def reset_chat(self):
        self.chat_history = []
