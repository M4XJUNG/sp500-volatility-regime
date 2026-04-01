"""
퀀트·ML 용어 대시보드 (최종 완성본)
실행: streamlit run app.py
"""

import streamlit as st
import duckdb
import pandas as pd
import os

# ─────────────────────────────────────────────
# 0. 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="퀀트·ML 용어 사전",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 1. CSS (글꼴 확대 및 한영 동일 선상 배치)
# ─────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { 
    font-family: 'Pretendard', 'Apple SD Gothic Neo', sans-serif; 
    font-size: 1.2rem; 
}

.term-card {
    background: #ffffff;
    border: 1px solid #e8eaed;
    border-radius: 12px;
    padding: 30px 40px; 
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    transition: box-shadow 0.2s;
}
.term-card:hover { box-shadow: 0 6px 16px rgba(0,0,0,0.15); }

.badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 18px; 
    font-weight: 600;
    margin-right: 10px;
    margin-bottom: 8px;
}
.badge-quant  { background:#E6F1FB; color:#1565C0; }
.badge-ml     { background:#E8F5E9; color:#2E7D32; }
.badge-sub    { background:#F3E5F5; color:#6A1B9A; }
.badge-week   { background:#FFF3E0; color:#E65100; }
.badge-proj   { background:#FCE4EC; color:#AD1457; }
.badge-order  { background:#E0F7FA; color:#006064; border: 1px solid #00ACC1; }

/* 🚀 한글 용어와 영문명을 같은 줄에 배치하기 위한 Flexbox 설정 */
.term-header {
    display: flex;
    align-items: baseline; /* 글자 밑단 기준 정렬 */
    gap: 15px; /* 한글과 영어 사이 간격 */
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.term-title { 
    font-size: 42px; 
    font-weight: 800; 
    color: #1a1a2e; 
    line-height: 1.2;
}
.term-en { 
    font-size: 24px; 
    color: #666; 
    font-family: monospace; 
}

.sec-label {
    font-size: 18px; 
    font-weight: 700;
    color: #888; 
    text-transform: uppercase;
    letter-spacing: 0.05em; 
    margin-bottom: 6px;
    margin-top: 20px;
}
.sec-text { 
    font-size: 28px; 
    color: #222; 
    line-height: 1.6; 
    margin-bottom: 15px; 
    font-weight: 500;
}
.formula-box {
    background: #f8f9fa;
    border-left: 5px solid #4A90E2;
    border-radius: 8px;
    padding: 15px 20px;
    font-family: monospace;
    font-size: 26px; 
    color: #111;
    margin-bottom: 15px;
}
.tip-box {
    background: #FFFDE7;
    border-left: 5px solid #F9A825;
    border-radius: 8px;
    padding: 15px 20px;
    font-size: 24px; 
    color: #444;
    margin-bottom: 10px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 2. DuckDB 초기화 
# ─────────────────────────────────────────────
DB_FILE  = "terms.duckdb"
CSV_FILE = "quant_ml_terms.csv"

@st.cache_resource
def get_db():
    con = duckdb.connect(DB_FILE)
    con.execute("""
        CREATE TABLE IF NOT EXISTS terms (
            id          INTEGER PRIMARY KEY,
            용어        TEXT NOT NULL,
            영문명      TEXT,
            대분류      TEXT,
            소분류      TEXT,
            한줄정의    TEXT,
            예시        TEXT,
            수식        TEXT,
            팁          TEXT,
            강의주차    TEXT,
            이해도      TEXT DEFAULT '미분류',
            프로젝트연결 BOOLEAN DEFAULT FALSE,
            추가일      DATE DEFAULT CURRENT_DATE
        )
    """)
    if os.path.exists(CSV_FILE):
        cnt = con.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
        if cnt == 0:
            tmp = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
            tmp.insert(0, 'id', range(1, len(tmp)+1))
            for col in ['이해도','프로젝트연결','추가일']:
                if col not in tmp.columns:
                    if col == '이해도':        tmp[col] = '미분류'
                    elif col == '프로젝트연결': tmp[col] = False
                    elif col == '추가일':       tmp[col] = pd.Timestamp.today().date()
            con.execute("INSERT INTO terms SELECT * FROM tmp")
    return con

con = get_db()


# ─────────────────────────────────────────────
# 3. 데이터 로드 함수 
# ─────────────────────────────────────────────
def load_data(where_clause="1=1", order_by="id ASC") -> pd.DataFrame:
    return con.execute(f"""
        SELECT * FROM terms
        WHERE {where_clause}
        ORDER BY {order_by}
    """).df()

def load_stats() -> dict:
    return {
        "total"   : con.execute("SELECT COUNT(*) FROM terms").fetchone()[0],
        "quant"   : con.execute("SELECT COUNT(*) FROM terms WHERE 대분류='퀀트·금융'").fetchone()[0],
        "ml"      : con.execute("SELECT COUNT(*) FROM terms WHERE 대분류='ML'").fetchone()[0],
        "proj"    : con.execute("SELECT COUNT(*) FROM terms WHERE 프로젝트연결=TRUE").fetchone()[0],
        "understood": con.execute("SELECT COUNT(*) FROM terms WHERE 이해도='완전이해'").fetchone()[0],
        "unknown" : con.execute("SELECT COUNT(*) FROM terms WHERE 이해도='모름'").fetchone()[0],
        "소분류"  : con.execute("SELECT DISTINCT 소분류 FROM terms ORDER BY 소분류").df()['소분류'].tolist(),
    }


# ─────────────────────────────────────────────
# 4. 사이드바 (검색 및 정렬 기능)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 용어 사전")
    stats = load_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("전체", stats["total"])
        st.metric("퀀트·금융", stats["quant"])
    with col2:
        st.metric("ML", stats["ml"])
        st.metric("프로젝트 연결", stats["proj"])

    st.divider()

    st.markdown("**🔍 통합 검색**")
    search_q = st.text_input("용어, 영문명, 정의 등 검색", placeholder="예: 샤프 비율, MLE...")

    st.markdown("**🔄 정렬 기준**")
    sort_option = st.selectbox(
        "정렬 방식을 선택하세요",
        ["학습 권장 순서 (기본)", "가나다순 (용어 기준)", "소분류 묶음 순"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("**대분류 필터**")
    cat_all   = st.checkbox("전체", value=True, key="cat_all")
    cat_quant = st.checkbox("퀀트·금융", value=False, key="cat_quant")
    cat_ml    = st.checkbox("ML", value=False, key="cat_ml")

    st.markdown("**소분류 필터**")
    subcats = ["(전체)"] + stats["소분류"]
    sel_sub = st.selectbox("소분류 선택", subcats, label_visibility="collapsed")

    st.markdown("**이해도 필터**")
    sel_understanding = st.selectbox(
        "이해도", ["전체","미분류","모름","보통","완전이해"],
        label_visibility="collapsed"
    )

    st.divider()
    view_mode = st.radio("보기 방식", ["카드 (크게 보기)", "표 (요약 보기)"], horizontal=True)


# ─────────────────────────────────────────────
# 5. SQL 조건 및 정렬 생성
# ─────────────────────────────────────────────
conditions = []

if search_q.strip():
    q = search_q.strip().replace("'", "''")
    conditions.append(f"""(
        용어 ILIKE '%{q}%' OR
        영문명 ILIKE '%{q}%' OR
        소분류 ILIKE '%{q}%' OR
        한줄정의 ILIKE '%{q}%' OR
        팁 ILIKE '%{q}%'
    )""")

if not cat_all:
    cats = []
    if cat_quant: cats.append("'퀀트·금융'")
    if cat_ml:    cats.append("'ML'")
    if cats:
        conditions.append(f"대분류 IN ({','.join(cats)})")

if sel_sub != "(전체)":
    s = sel_sub.replace("'", "''")
    conditions.append(f"소분류 = '{s}'")

if sel_understanding != "전체":
    u = sel_understanding.replace("'", "''")
    conditions.append(f"이해도 = '{u}'")

where = " AND ".join(conditions) if conditions else "1=1"

order_by_clause = "CAST(id AS INTEGER) ASC"
if sort_option == "가나다순 (용어 기준)":
    order_by_clause = "용어 ASC"
elif sort_option == "소분류 묶음 순":
    order_by_clause = "소분류 ASC, CAST(id AS INTEGER) ASC"

df = load_data(where, order_by_clause)


# ─────────────────────────────────────────────
# 6. 메인 화면
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 용어 학습", "➕ 용어 관리", "📊 학습 통계"])

# ════════════════════════════════
# TAB 1: 용어 학습
# ════════════════════════════════
with tab1:
    hcol1, hcol2 = st.columns([3, 1])
    with hcol1:
        st.markdown(f"### 🎯 현재 조건에 맞는 용어: **{len(df)}개**")
    
    if df.empty:
        st.info("검색 결과가 없습니다. 필터나 검색어를 조정해 보세요.")
    else:
        if view_mode == "카드 (크게 보기)":
            BADGE_COLOR = {"퀀트·금융": "badge-quant", "ML": "badge-ml"}
            
            def safe_text(val):
                if pd.isna(val) or str(val).strip() in ["nan", "None", ""]:
                    return ""
                return str(val).strip()

            for index, row in df.iterrows():
                badge_cat = BADGE_COLOR.get(row["대분류"], "badge-sub")
                order_badge = f'<span class="badge badge-order">📖 Step {row["id"]}</span>'
                
                week_val = safe_text(row.get("강의주차"))
                week_badge = f'<span class="badge badge-week">📅 {week_val}</span>' if week_val else ""

                formula_val = safe_text(row.get("수식"))
                if formula_val and formula_val != "없음":
                    formula_html = f"""<div class="sec-label">수식</div>
<div class="formula-box">

{formula_val}

</div>"""
                else:
                    formula_html = ""

                tip_val = safe_text(row.get("팁"))
                tip_html = f'<div class="tip-box">💡 <strong>실무/팁:</strong> {tip_val}</div>' if tip_val else ""

                term_title = safe_text(row['용어'])
                term_en = safe_text(row.get('영문명'))
                term_def = safe_text(row['한줄정의'])
                
                term_ex = safe_text(row.get('예시'))
                ex_html = f'<div class="sec-label">예시</div><div class="sec-text">{term_ex}</div>' if term_ex else ""
                
                # 영문명이 있을 때만 태그 생성
                term_en_html = f'<span class="term-en">{term_en}</span>' if term_en else ""

                st.markdown(f"""<div class="term-card">
{order_badge}
<span class="badge {badge_cat}">{safe_text(row['대분류'])}</span>
<span class="badge badge-sub">{safe_text(row['소분류'])}</span>
{week_badge}
<br><br>
<div class="term-header">
    <span class="term-title">{term_title}</span>
    {term_en_html}
</div>
<div class="sec-label">한 줄 정의</div>
<div class="sec-text">{term_def}</div>
{ex_html}
{formula_html}
{tip_html}
</div>""", unsafe_allow_html=True)

        elif view_mode == "표 (요약 보기)":
            show_cols = ["id", "용어", "영문명", "대분류", "소분류", "한줄정의"]
            st.dataframe(
                df[show_cols],
                use_container_width=True,
                height=800,
                hide_index=True
            )

# ════════════════════════════════
# TAB 2: 용어 관리 (활성화 복구)
# ════════════════════════════════
with tab2:
    st.markdown("### ➕ 새로운 용어 추가")
    with st.form("add_term_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_term = st.text_input("용어 (한글) *", max_chars=100)
            new_cat = st.selectbox("대분류 *", ["퀀트·금융", "ML", "기타"])
            new_def = st.text_area("한 줄 정의 *", height=100)
        with col2:
            new_en = st.text_input("영문명", max_chars=100)
            new_subcat = st.text_input("소분류", placeholder="예: 통계, 회귀, 지표 등")
            new_week = st.text_input("강의주차", placeholder="예: 1주차")
            
        new_ex = st.text_area("예시", height=100)
        new_formula = st.text_input("수식 (평문 입력)")
        new_tip = st.text_input("실무/팁")
        
        submit_btn = st.form_submit_button("용어 추가하기")
        
        if submit_btn:
            if not new_term or not new_def:
                st.error("용어와 한 줄 정의는 필수 입력 항목입니다.")
            else:
                new_id = con.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM terms").fetchone()[0]
                con.execute("""
                    INSERT INTO terms (id, 용어, 영문명, 대분류, 소분류, 한줄정의, 예시, 수식, 팁, 강의주차)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (new_id, new_term, new_en, new_cat, new_subcat, new_def, new_ex, new_formula, new_tip, new_week))
                st.success(f"'{new_term}' 용어가 성공적으로 추가되었습니다! 새로고침(F5)을 눌러 확인하세요.")

# ════════════════════════════════
# TAB 3: 학습 통계
# ════════════════════════════════
with tab3:
    st.markdown("### 📊 학습 통계 대시보드")
    col1, col2, col3 = st.columns(3)
    col1.metric("총 등록 용어", stats["total"])
    col2.metric("퀀트·금융", stats["quant"])
    col3.metric("머신러닝(ML)", stats["ml"])