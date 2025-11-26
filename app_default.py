# define gemini api key
import os 
# ⚠️ GÜVENLİK UYARISI: API Key'inizi buraya yapıştırın.
os.environ["GEMINI_API_KEY"] = "AIzaSyAPLOu_CX_73zmAeZ2Ia_asWrB7B_7BADU"

# import the necessary library
import time
import sys
import fitz
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Streamlit sayfa ayarları
st.set_page_config(
    page_title="Gazi AI Asistan", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 🎨 GELİŞMİŞ WHATSAPP TARZI & MODERN UI TASARIMI ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* GENEL SAYFA YAPISI */
    .stApp {
        background-color: #0b141a; /* WhatsApp Dark Arkaplanı */
        font-family: 'Inter', sans-serif;
    }

    /* --- SOHBET BALONLARI (WHATSAPP TARZI) --- */
    
    /* 1. KULLANICI MESAJI (SAĞDA) */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse; /* Avatarı ve balonu ters çevir */
        text-align: right;
    }
    
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #005c4b 0%, #008069 100%); /* WhatsApp Yeşili */
        color: #e9edef;
        border-radius: 15px 0px 15px 15px; /* Sağ üst köşe sivri */
        padding: 10px 16px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        border: none;
        display: inline-block;
        text-align: left; /* Metin balon içinde soldan başlasın */
        max-width: 80%; /* Balon çok uzamasın */
    }

    /* 2. ASİSTAN MESAJI (SOLDA) */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        flex-direction: row;
        text-align: left;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stChatMessageContent"] {
        background-color: #202c33; /* WhatsApp Koyu Gri */
        color: #e9edef;
        border-radius: 0px 15px 15px 15px; /* Sol üst köşe sivri */
        padding: 10px 16px;
        border: 1px solid #30363d;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        display: inline-block;
        max-width: 85%;
    }

    /* AVATARLAR */
    [data-testid="stChatMessageAvatarUser"] {
        background-color: #00a884;
        margin-left: 10px;
    }
    
    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: #202c33;
        margin-right: 10px;
        border: 1px solid #37404a;
    }

    /* --- MODERN SIDEBAR (SOL MENÜ) --- */
    
    [data-testid="stSidebar"] {
        background-color: #111b21;
        border-right: 1px solid #222d36;
    }
    
    /* Sidebar Başlık Kartı */
    .sidebar-header {
        text-align: center;
        padding: 20px 10px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Sidebar İstatistik Kartı */
    .stat-card {
        background: linear-gradient(145deg, #1f2937, #111827);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        margin-bottom: 15px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .stat-number {
        font-size: 28px;
        font-weight: 700;
        color: #3b82f6; /* Gazi Mavisi */
        margin: 5px 0;
    }
    
    .stat-label {
        font-size: 12px;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Buton Tasarımı */
    .stButton button {
        background-color: #ef4444;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        background-color: #dc2626;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }

    /* Input Alanı (Chat Bar) */
    .stChatInput {
        padding-bottom: 1rem;
    }
    
    .stChatInput input {
        background-color: #2a3942 !important;
        border: 1px solid #2a3942 !important;
        color: white !important;
        border-radius: 24px !important;
    }
    
    .stChatInput input:focus {
        border-color: #00a884 !important;
    }

    /* Header Alanı */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, rgba(0,0,0,0) 70%);
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        background: linear-gradient(to right, #60a5fa, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #111b21; 
    }
    ::-webkit-scrollbar-thumb {
        background: #37404a; 
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Cache ile veritabanını yükle (MANTIĞA DOKUNULMADI)
@st.cache_resource
def load_vector_store():
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    DB_PATH = "./gazi_db_hf"
    DATA_PATH = "./data"
    
    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        vector_store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=hf_embeddings,
        )
        return vector_store
    else:
        if not os.path.exists(DATA_PATH):
            st.error(f"❌ '{DATA_PATH}' klasörü bulunamadı!")
            sys.exit()

        docs = []
        pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]

        if not pdf_files:
            st.error("❌ 'data' klasöründe PDF bulunamadı.")
            sys.exit()

        for filename in pdf_files:
            file_path = os.path.join(DATA_PATH, filename)
            try:
                doc = fitz.open(file_path)
                for i, page in enumerate(doc):
                    text = page.get_text("text")
                    if text.strip():
                        docs.append(Document(
                            page_content=text,
                            metadata={"source": file_path, "page": i}
                        ))
            except Exception as e:
                continue

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["MADDE", "Madde", "\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(docs)

        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=hf_embeddings,
            persist_directory=DB_PATH,
        )
        return vector_store

# LLM ve Chain'leri cache'le (MANTIĞA DOKUNULMADI)
@st.cache_resource
def load_rag_chain(_vector_store):
    retriever = _vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5}
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    
    template = """Sen Gazi Üniversitesi mevzuat asistanısın. Görevin aşağıdaki bağlamı kullanarak soruları cevaplamaktır.

KURALLAR:
1. Sadece verilen BAĞLAM içindeki bilgileri kullan
2. Cevabı oluştururken ilgili yönetmelik maddesini bul ve açıkla
3. Eğer soru bir prosedürse, adımları sıralı şekilde yaz
4. Bilgi bağlamda yoksa "Bu bilgi dokümanlarımda yer almıyor" de
5. Cevabını doğrudan ver, gereksiz başlıklar kullanma

BAĞLAM:
{context}

SORU: {question}

CEVAP:"""
    
    prompt = PromptTemplate.from_template(template=template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )
    
    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    
    return rag_chain_with_source

# Veritabanı yüklemesi
with st.spinner("🚀 Sistem başlatılıyor, veritabanı taranıyor..."):
    vector_store = load_vector_store()
    rag_chain = load_rag_chain(vector_store)

# Header (Ana Başlık)
st.markdown("""
<div class="main-header">
    <h1>🏛️ Gazi AI Asistan</h1>
    <p style="color: #8696a0;">Akademik Mevzuat & Yönetmelik Rehberi</p>
</div>
""", unsafe_allow_html=True)

# Chat geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- MODERN SIDEBAR TASARIMI ---
with st.sidebar:
    # Logo / Başlık Alanı
    st.markdown("""
    <div class="sidebar-header">
        <div style="font-size: 50px; margin-bottom: 10px;">🤖</div>
        <h3 style="color: white; margin: 0;">Gazi Asistan</h3>
        <p style="color: #6b7280; font-size: 12px; margin-top: 5px;">Mevzuat & Yönetmelik Botu v2.2</p>
    </div>
    """, unsafe_allow_html=True)
    
    # İstatistik Kartı
    total_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{total_messages}</div>
        <div class="stat-label">Toplam Sorgu</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Bilgilendirme
    with st.expander("ℹ️ Sistem Hakkında"):
        st.caption("Bu sistem RAG (Retrieval-Augmented Generation) mimarisi kullanılarak Gazi Üniversitesi mevzuatlarını tarar.")

    st.markdown("---")
    
    # Temizle Butonu
    if st.button("🗑️ Sohbeti Sıfırla", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("""
    <div style="position: fixed; bottom: 10px; font-size: 10px; color: #4b5563; text-align: center; width: 250px;">
        Gazi AI Core Systems © 2025
    </div>
    """, unsafe_allow_html=True)

# Geçmiş mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# Kullanıcı input
if prompt := st.chat_input("Mevzuat hakkında merak ettiklerinizi sorun..."):
    # Kullanıcı mesajı
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Bot cevabı
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with st.spinner("🔍 Mevzuat taranıyor..."):
                sonuc = rag_chain.invoke(prompt)
                cevap_metni = sonuc['answer']
                dokumanlar = sonuc['context']
            
            # Temizlik
            cevap_metni = cevap_metni.replace("## ✅ Cevap:", "").replace("## 📌 KURALLAR", "").strip()
            
            # Typewriter efekti (Yazıyormuş gibi)
            for char in cevap_metni:
                full_response += char
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.005)
            
            message_placeholder.markdown(full_response)
            
            # Kaynaklar (Expander içinde)
            if len(cevap_metni) > 20 and "yer almıyor" not in cevap_metni.lower():
                with st.expander("📚 Referans Dokümanlar"):
                    seen = set()
                    for doc in dokumanlar:
                        source_name = os.path.basename(doc.metadata.get("source", "Bilinmiyor"))
                        page = doc.metadata.get("page", "-")
                        try:
                            visible_page = int(page) + 1
                        except:
                            visible_page = page
                        
                        unique_id = f"{source_name}-{page}"
                        if unique_id not in seen:
                            st.markdown(f"- 📄 `{source_name}` (Sayfa: {visible_page})")
                            seen.add(unique_id)
            
        except Exception as e:
            full_response = f"⚠️ Bir hata oluştu: {str(e)}"
            message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})