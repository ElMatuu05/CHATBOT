import streamlit as st
from bot import UNLPamComputacionChatbot

st.set_page_config(page_title="ChatBot matHIAs", page_icon="💻", layout="wide")

# =========================
# Estilos CSS personalizados
# =========================
st.markdown(
    """
    <style>
    
    /* Ocultar la barra superior de Streamlit */
    header[data-testid="stHeader"] {
        display: none;
    }

    /* Ocultar el menú superior derecho (los tres puntitos) */
    [data-testid="stToolbar"] {
        display: none;
    }

    /* Fondo general */
    body {
        background-color: #f5f9ff;
    }
    /* Encabezado fijo */
    .header {
        position: fixed;
        top: 0; left: 0; right: 0;
        background: linear-gradient(90deg, #004080, #007f66);
        color: white;
        padding: 15px 30px;
        height: 70px;
        z-index: 999;
        border-bottom: 4px solid #ff6600;
        display: flex;
        align-items: center;
    }
    .header img {
        margin-right: 15px;
        border-radius: 6px;
    }
    .header h2 {
        margin: 0;
        font-size: 24px;
    }
    .spacer {
        margin-top: 90px;
    }
    /* Botones FAQ */
    div[data-testid="stButton"] > button {
        background-color: #ff6600;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 10px 16px;
        font-weight: bold;
        transition: 0.3s;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #cc5200;
        transform: scale(1.05);
    }
    /* Mensajes del chat */
    .stChatMessage.user {
        background-color: #e6f0ff;
        border: 1px solid #00408033;
        border-radius: 12px;
        padding: 10px;
    }
    .stChatMessage.assistant {
        background-color: #eafff5;
        border: 1px solid #007f6633;
        border-radius: 12px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Inicialización
# =========================
if "chatbot" not in st.session_state:
    st.session_state.chatbot = UNLPamComputacionChatbot(
        pdf_folder="data",
        urls=[
            "https://exactas.unlpam.edu.ar/academica/bedelia/horarios",
            "https://exactas.unlpam.edu.ar/carreras/profesorados/profesorado-en-computacion/",
            "https://exactas.unlpam.edu.ar/estudiantes/ingresantes-2025/",
            "https://exactas.unlpam.edu.ar/carreras/profesorados/profesorado-en-computacion/plan-de-estudios-2015/?utm_source=chatgpt.com",
            "https://exactas.unlpam.edu.ar/wp-content/uploads/2024/09/Profesorado-en-Computacion-2015.pdf?utm_source=chatgpt.com",
            "https://exactas.unlpam.edu.ar/carreras/profesorados/profesorado-en-computacion/?utm_source=chatgpt.com",
            "https://exactas.unlpam.edu.ar/academica/bedelia/?utm_source=chatgpt.com",
            "https://exactas.unlpam.edu.ar/academica/bedelia/horarios/?utm_source=chatgpt.com",
            "https://exactas.unlpam.edu.ar/wp-content/uploads/2024/05/392-15-Prof.-Computacion.pdf?utm_source=chatgpt.com",
        ]
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "past_conversations" not in st.session_state:
    st.session_state.past_conversations = []

# =========================
# Encabezado fijo con logo y título
# =========================
st.markdown(
    """
    <div class="header">
        <img src="https://exactas.unlpam.edu.ar/templates/yootheme/cache/facultad-ciencias-exactas-naturales-logo-6b65c7ac.png" 
             alt="Logo Exactas UNLPam" width="55">
        <h2>💻 ChatBot mathIA's</h2>
    </div>
    <div class="spacer"></div>
    """,
    unsafe_allow_html=True
)

# =========================
# Botones de FAQs fijos
# =========================
st.markdown(
    "<h1 style='text-align: center; color: #004080;'>mathIA's el bot que responde sobre el Profesorado en Computación</h1>",
    unsafe_allow_html=True
)

st.markdown("### ❓ Preguntas rápidas")
faq_col1, faq_col2, faq_col3, faq_col4 = st.columns(4)

with faq_col1:
    if st.button("📚 Duración"):
        pregunta = "¿Cuántos años dura el profesorado en computación?"
        respuesta = st.session_state.chatbot.get_answer(pregunta)
        st.session_state.messages.append(("user", pregunta))
        st.session_state.messages.append(("assistant", respuesta))

with faq_col2:
    if st.button("🎓 Título"):
        pregunta = "¿Qué título otorga el profesorado en computación?"
        respuesta = st.session_state.chatbot.get_answer(pregunta)
        st.session_state.messages.append(("user", pregunta))
        st.session_state.messages.append(("assistant", respuesta))

with faq_col3:
    if st.button("📝 Materias"):
        pregunta = "¿Qué materias tiene el profesorado en computación?"
        respuesta = st.session_state.chatbot.get_answer(pregunta)
        st.session_state.messages.append(("user", pregunta))
        st.session_state.messages.append(("assistant", respuesta))

with faq_col4:
    if st.button("🏫 Modalidad"):
        pregunta = "¿La cursada es presencial o virtual?"
        respuesta = st.session_state.chatbot.get_answer(pregunta)
        st.session_state.messages.append(("user", pregunta))
        st.session_state.messages.append(("assistant", respuesta))

# =========================
# Mostrar historial de chat
# =========================
for role, text in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(text)

# =========================
# Primer mensaje automático
# =========================
if not st.session_state.messages:
    bienvenida = "¡Hola 👋! Soy **ChatBot matHIAs** del Profesorado en Computación de la UNLPam. Preguntame lo que quieras sobre la carrera 🤓."
    st.session_state.messages.append(("assistant", bienvenida))
    with st.chat_message("assistant"):
        st.markdown(bienvenida)

# =========================
# Entrada de usuario
# =========================
if prompt := st.chat_input("Escribí tu consulta..."):
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            respuesta = st.session_state.chatbot.get_answer(prompt)
        except Exception as e:
            respuesta = f"❌ Error: {str(e)}"
        st.markdown(respuesta)

    st.session_state.messages.append(("assistant", respuesta))

# =========================
# Barra lateral: Reset + Historial
# =========================
if st.sidebar.button("🔄 Resetear conversación"):
    if st.session_state.messages:
        st.session_state.past_conversations.append(st.session_state.messages.copy())
    st.session_state.chatbot.reset_chat()
    st.session_state.messages = []
    st.sidebar.success("La conversación fue reseteada y guardada en el historial.")

if st.session_state.past_conversations:
    st.sidebar.markdown("## 💾 Conversaciones pasadas")
    for i, conv in enumerate(st.session_state.past_conversations[::-1], 1):
        if st.sidebar.button(f"📂 Chat {len(st.session_state.past_conversations)-i+1}"):
            st.session_state.messages = conv.copy()
