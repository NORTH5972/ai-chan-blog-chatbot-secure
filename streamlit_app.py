import streamlit as st
import json
import re
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import bcrypt 

# --- 1. Google Gemini API キーの設定 ---
API_KEY = os.getenv("GOOGLE_API_KEY") 
if not API_KEY and "GOOGLE_API_KEY" in st.secrets: 
    API_KEY = st.secrets["GOOGLE_API_KEY"]

if not API_KEY:
    st.error("エラー: Google Gemini API キーが設定されていません。")
    st.error("Streamlit CloudのSecretsまたは環境変数に 'GOOGLE_API_KEY' を設定してください。")
    st.stop() 

genai.configure(api_key=API_KEY)

# === 共通の埋め込みモデルとLLMモデルのロード関数 (st.cache_resource でキャッシュ) ===
@st.cache_resource
def get_llm_model():
    st.info("LLMモデルをロード中...")
    model = genai.GenerativeModel('models/gemini-1.5-pro-latest') 
    st.success("LLMモデルのロードが完了しました。")
    return model

@st.cache_resource
def get_query_embedding_model(model_name="intfloat/multilingual-e5-small"):
    st.info("クエリ埋め込みモデルをロード中...")
    model = SentenceTransformer(model_name)
    st.success("クエリ埋め込みモデルのロードが完了しました。")
    return model

# === 記事データの読み込みとクリーンアップ関数 ===
def load_and_clean_data(file_path):
    cleaned_articles = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        for article in articles:
            body = article.get('body', '') 
            body = re.sub(r'http://googleusercontent\.com/youtube\.com/\d+', '', body)
            body = re.sub(r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\.(?:jpg|jpeg|png|gif|bmp|webp)', '', body) 
            body = re.sub(r'﻿', '', body) 
            body = re.sub(r'\n\s*\n', '\n', body) 
            body = re.sub(r'\s{2,}', ' ', body)   
            body = body.strip()
            if len(body) < 20: 
                continue
            cleaned_articles.append({
                'title': article.get('title'),
                'date': article.get('date'),
                'url': article.get('url'),
                'body': body 
            })
        return cleaned_articles
    except FileNotFoundError:
        st.error(f"エラー: ファイル '{file_path}' が見つかりませんでした。ファイルパスを確認してください。")
        return []
    except json.JSONDecodeError:
        st.error(f"エラー: ファイル '{file_path}' が不正なJSON形式です。")
        return []
    except Exception as e:
        st.error(f"データの読み込みまたはクリーンアップ中に予期せぬエラーが発生しました: {e}")
        return []

# === 埋め込みモデルのロードと記事のベクトル化 ===
@st.cache_resource
def load_or_generate_embeddings(articles, _embedding_model_instance):
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(METADATA_FILE):
        try:
            existing_embeddings = np.load(EMBEDDINGS_FILE)
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)
            
            if len(existing_metadata) == len(articles) and existing_embeddings.shape[0] == len(articles):
                st.info("既存の埋め込みデータと記事データ数が一致します。再生成をスキップします。")
                return existing_embeddings, existing_metadata
            else:
                st.warning("既存の埋め込みデータと記事データ数が一致しないため、埋め込みを再生成します。")
        except Exception as e:
            st.warning(f"既存の埋め込みファイルのロード中にエラーが発生しました ({e})。埋め込みを再生成します。")

    st.info("埋め込みベクトルを生成中... (初回はモデルダウンロードが発生します)")
    
    corpus = [article['body'] for article in articles]
    embeddings = _embedding_model_instance.encode(corpus, batch_size=32, show_progress_bar=False) 
    
    st.info("ベクトル化が完了しました。")

    metadata = []
    for i, article in enumerate(articles): 
        metadata.append({
            'original_index': i, 
            'title': article['title'], 
            'url': article['url'], 
            'date': article['date'], 
            'body_preview': article['body'][:100].replace('\n', ' ').strip() + '...' if len(article['body']) > 100 else article['body'].strip()
        })
    
    np.save(EMBEDDINGS_FILE, embeddings)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    
    st.success(f"埋め込みベクトルを '{EMBEDDINGS_FILE}' に、メタデータを '{METADATA_FILE}' に保存しました。")
    
    return embeddings, metadata

# === RAGChatbotクラス ===
class RAGChatbot:
    def __init__(self, embeddings, metadata, articles_data, llm_model, query_embedding_model):
        self.embeddings = embeddings 
        self.metadata = metadata     
        self.articles_data = articles_data 
        
        self.llm_model = llm_model 
        self.query_embedding_model = query_embedding_model 

    def find_relevant_articles(self, query, top_k=3):
        """
        質問の埋め込みベクトルを生成し、最も類似度の高い記事を検索します。
        """
        query_embedding = self.query_embedding_model.encode([query])[0]
        
        norms_embeddings = np.linalg.norm(self.embeddings, axis=1)
        norm_query = np.linalg.norm(query_embedding)
        
        similarities = np.zeros(len(self.embeddings))
        non_zero_indices = np.where(norms_embeddings != 0)[0]
        if norm_query != 0:
            similarities[non_zero_indices] = np.dot(self.embeddings[non_zero_indices], query_embedding) / (norms_embeddings[non_zero_indices] * norm_query)
        
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        
        relevant_articles = []
        for i_metadata in top_k_indices:
            original_article_index = self.metadata[i_metadata]['original_index']
            if 0 <= original_article_index < len(self.articles_data):
                relevant_articles.append(self.articles_data[original_article_index])
            else:
                st.warning(f"警告: original_index {original_article_index} がarticles_dataの範囲外です。")
            
        return relevant_articles

    def generate_response(self, query, relevant_articles):
        """
        検索で得られた関連記事と質問をLLMに渡し、回答を生成させます。
        """
        context = ""
        if relevant_articles:
            context += "以下のブログ記事の情報を参照して質問に答えてください。\n"
            for i, article in enumerate(relevant_articles):
                context += f"--- 記事 {i+1} (タイトル: {article['title']}, 日付: {article['date']}, URL: {article['url']}) ---\n"
                context += article['body'] + "\n\n"
            context += "--- 参照情報終わり ---\n\n"
            context += "参照情報に基づいて質問に答えてください。参照情報にない場合は、「申し訳ありません、その情報が見つかりませんでした。」と答えてください。\n"
            context += "回答の最後に、参照した記事の番号、タイトル、日付、URLを必ず記載してください。\n"
            context += "例：\n"
            context += "回答：はい、〇〇です。記事1（タイトル：XXX、日付：YYYY.MM.DD、URL：ZZZ）をご覧ください。\n"
            context += "加えて、〇〇の点も考慮してください。\n" 
        else:
            context += "参照できるブログ記事情報がありません。質問に答えることができません。\n"
            return "申し訳ありません、その情報が見つかりませんでした。別の質問をお試しください。"

        prompt = f"{context}質問: {query}\n回答:"
        
        try:
            response = self.llm_model.generate_content(prompt)
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text.strip()
            else:
                return "LLMからの回答生成に問題が発生しました。再度お試しください。"
        except Exception as e:
            st.error(f"LLMからの回答生成中にエラーが発生しました: {e}")
            if "blocked_reason" in str(e): 
                return "申し訳ありません、この質問はコンテンツポリシーに違反する可能性があります。"
            elif "Quota exceeded" in str(e): 
                return "申し訳ありません、APIの利用制限に達しました。しばらく時間をおいてからお試しください。"
            return "現在、回答を生成できません。時間をおいて再度お試しください。 (APIエラー)"

# --- 6. Streamlit アプリケーションのUIとロジック（認証部分を先頭に） ---

# ファイル名と定数を定義
DATA_FILE = "endo_sakura_blog_data.json"
EMBEDDINGS_FILE = "blog_embeddings.npy"
METADATA_FILE = "blog_metadata.json"

st.set_page_config(page_title="愛ちゃんブログチャットボット", page_icon="🌸")

# --- 手動認証情報の定義 ---
# Streamlit Secretsから直接、ユーザー名とハッシュ化パスワードを読み込む
VALID_USERNAME = "user" # secrets.tomlで設定したユーザー名と一致させる
VALID_HASHED_PASSWORD = st.secrets.get("auth", {}).get("credentials", {}).get("usernames", {}).get(VALID_USERNAME, {}).get("password")

# 認証されていない場合のデフォルト値
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

# --- 認証UIの表示 ---
if not st.session_state["logged_in"]:
    st.header("ログインしてください")
    username_input = st.text_input("ユーザー名")
    password_input = st.text_input("パスワード", type="password")

    if st.button("ログイン"):
        if VALID_HASHED_PASSWORD:
            try:
                # 入力されたパスワードをハッシュ化して比較
                # bcrypt.checkpw はバイト文字列を比較するため、入力とハッシュ値をエンコード
                if bcrypt.checkpw(password_input.encode('utf-8'), VALID_HASHED_PASSWORD.encode('utf-8')):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username_input
                    st.success(f"ようこそ、{username_input}さん！")
                    st.rerun() # ログイン成功時にページを再描画してチャットボットを表示 <-- ここを修正
                else:
                    st.error("ユーザー名またはパスワードが間違っています。")
            except ValueError: # ハッシュ値の形式が不正な場合など
                st.error("認証情報の設定に問題があります。管理者に連絡してください。")
        else:
            st.error("認証情報が正しく設定されていません。")
else: # 認証済みの場合
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"ようこそ、{st.session_state['username']}さん！")
    if st.sidebar.button("ログアウト"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.rerun() # ログアウト時にページを再描画してログインフォームを表示 <-- ここを修正

    st.title("🌸 愛ちゃんブログチャットボット")
    st.write("遠藤さくらさんのブログ記事から情報を取得して、質問に答えます。")

    @st.cache_resource
    def initialize_chatbot_instance(): 
        with st.spinner("記事データを読み込み、クリーンアップ中..."):
            articles_data = load_and_clean_data(DATA_FILE)
            if not articles_data:
                st.error("記事データが読み込めませんでした。`endo_sakura_blog_data.json` が存在するか確認してください。")
                return None
            st.success(f"クリーンアップされた記事データ: {len(articles_data)}件")

        llm_model_instance = get_llm_model()
        query_embedding_model_instance = get_query_embedding_model()

        with st.spinner("埋め込みベクトルを生成またはロード中..."):
            embeddings, metadata = load_or_generate_embeddings(articles_data, query_embedding_model_instance) 
            if embeddings is None or len(embeddings) == 0:
                st.error("埋め込みベクトルが生成またはロードできませんでした。")
                return None
            st.success(f"埋め込みの形状: {embeddings.shape}, メタデータの件数: {len(metadata)}")

        with st.spinner("RAGチャットボットをセットアップ中..."):
            chatbot_instance = RAGChatbot(embeddings, metadata, articles_data, llm_model_instance, query_embedding_model_instance) 
            st.success("チャットボットの準備が整いました！")
        return chatbot_instance

    chatbot = initialize_chatbot_instance() 

    if chatbot is None:
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("質問を入力してください"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("回答を生成中..."):
                relevant_articles = chatbot.find_relevant_articles(prompt, top_k=3)
                response = chatbot.generate_response(prompt, relevant_articles)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    st.sidebar.markdown("---")
    st.sidebar.markdown("開発者情報")
    st.sidebar.markdown("このチャットボットは、Python, Streamlit, Sentence-Transformers, Google Gemini API を使用して構築されています。")