import sys, os, time, hashlib, re 
import yfinance as yf
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
except NameError:
    pass 
    
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler 
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from utils.chatbot_ai import get_ai_reply, fetch_stock_details, analyze_trend, plot_stock_chart
except Exception:
    try:
        from openai import OpenAI
    except ImportError:
        st.error("The 'openai' library is not installed. Chatbot will not work. Please run: pip install openai")
        class OpenAI:
            def __init__(self, *args, **kwargs):
                pass
            def chat(self, *args, **kwargs):
                return None

    def get_ai_reply(prompt):
        
        ticker_match = re.search(r'\b([A-Z0-9\-\.&]+)\.(NS|BO)\b', prompt.upper())
        
        if ticker_match:
            symbol = ticker_match.group(0)
            try:
                with st.spinner(f"Fetching {symbol} insights..."):
                    data = fetch_stock_details(symbol)
                    if not data or data["history"].empty:
                        return {"type": "text", "content": f"Sorry, I couldn't fetch data for {symbol}."}
                    
                    commentary, tone = analyze_trend(data["history"])
                    fig = plot_stock_chart(data["history"], symbol)
                    
                    price = data.get('price', 0)
                    name = data.get('name', symbol)
                    
                    return {
                        "type": "stock",
                        "content": f"Here's the analysis for **{name} ({symbol})**:\n"
                                   f"* **Current Price:** ₹{price:,.2f}\n"
                                   f"* **Trend:** {tone}\n"
                                   f"* **Analysis:** {commentary}",
                        "chart": fig
                    }
            except Exception as e:
                return {"type": "text", "content": f"Sorry, I had an error analyzing {symbol}: {e}"}

        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if not client.api_key:
                return {"type": "text", "content": "⚠️ OpenAI API key is missing. Please set it in your .env file."}
                
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a friendly Indian finance chatbot."},
                          {"role": "user", "content": prompt}],
            )
            return {"type": "text", "content": response.choices[0].message.content.strip()}
        except Exception as e:
            if "AuthenticationError" in str(e):
                 return {"type": "text", "content": "⚠️ OpenAI API key is incorrect or invalid."}
            return {"type": "text", "content": f"⚠️ Chatbot error: {e}"}

    def fetch_stock_details(symbol):
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="6mo", auto_adjust=True)
        if hist.empty:
            return None
        return {
            "symbol": symbol,
            "name": info.get("shortName", symbol),
            "sector": info.get("sector", "Unknown"),
            "price": hist["Close"].iloc[-1],
            "high": info.get("fiftyTwoWeekHigh", 0),
            "low": info.get("fiftyTwoWeekLow", 0),
            "history": hist,
        }

    def analyze_trend(df):
        df_copy = df.copy() 
        df_copy["change"] = df_copy["Close"].pct_change()
        avg_5 = df_copy["change"].tail(5).mean()
        avg_20 = df_copy["change"].tail(20).mean()
        if avg_5 > 0.01:
            tone = "Uptrend 📈"
        elif avg_5 < -0.01:
            tone = "Downtrend 📉"
        else:
            tone = "Neutral ⚖️"
        return f"Momentum over last 5 days: {avg_5*100:.2f}% vs 20D avg {avg_20*100:.2f}%", tone

    def plot_stock_chart(df, symbol):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", line=dict(width=2)))
        fig.update_layout(title=f"{symbol} — 6M Trend", template="plotly_dark", height=400)
        return fig

st.set_page_config(page_title="FINSIGHTS AI", page_icon="📊", layout="wide")
load_dotenv() 

st.markdown("""
<style>
body { background-color:#0E1117; color:#E5E5EE; } 
.main { background-color:#0E1117; }
h1,h2,h3 { color:#E5E5E5; }
.card { 
    background:rgba(255,255,255,0.04); 
    border-radius:12px; 
    padding:16px; 
    margin-bottom:10px; 
    border: 1px solid rgba(255, 255, 255, 0.1); 
    height: 100%; 
}
.stock-card { background-color:#121826; border-radius:15px; padding:16px; margin-top:10px; }
.stock-title { color:#00c6ff; font-weight:700; font-size:1.2rem; }
.stock-detail { color:#E5E5E5; font-size:0.95rem; line-height:1.6; }

.learn-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #00c6ff; 
    border-bottom: 2px solid #00c6ff;
    padding-bottom: 5px;
    margin-bottom: 15px;
    margin-top: 15px; 
}
.st-expander {
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    background-color: rgba(255, 255, 255, 0.02) !important;
}
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

menu_items = [
    "🏠 Home", "💬 Chat with AarthAI", "📊 Stock Insights",
    "📈 Predict Stock Trends", "💼 Portfolio", "🧮 Calculators",
    "🧭 Planner", "🎯 Side Investments", "📚 Learn"
]
choice = st.sidebar.radio("📍 Navigate", menu_items, index=menu_items.index(st.session_state.page))
if choice != st.session_state.page:
    st.session_state.page = choice
    st.rerun()

def rupee_fmt(v): return f"₹{v:,.0f}"

def page_home():
    st.title("📊 FINSIGHTS AI")
    st.subheader("Smarter Money. Simpler Future.")
    st.markdown("""
    FINSIGHTS is an AI-powered India-first finance companion for Gen-Z investors.  
    *Analyze stocks, plan your finances, predict trends, and learn — all in one place.*
    """)
    
    st.markdown("---")
    st.subheader("🚀 Key Features")
    
    c1, c2, c3 = st.columns(3)
    with c1: 
        st.markdown('<div class="card">🧭 <b>Financial Planner</b><br>Smart income & goal-based allocation for all your needs.</div>', unsafe_allow_html=True)
    with c2: 
        st.markdown('<div class="card">📊 <b>Stock Insights</b><br>AI-analyzed charts & 52W trend detection.</div>', unsafe_allow_html=True)
    with c3: 
        st.markdown('<div class="card">💬 <b>AarthAI Chatbot</b><br>Your personal Indian finance assistant. Ask it to "analyze RELIANCE.NS"!</div>', unsafe_allow_html=True)
    
    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown('<div class="card">📈 <b>Trend Prediction</b><br>Uses Polynomial Regression to forecast future stock prices.</div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="card">📚 <b>Learn Academy</b><br>Bite-sized lessons on stocks, funds, and more.</div>', unsafe_allow_html=True)
    with c6:
        st.markdown('<div class="card">🎯 <b>Side Investments</b><br>Explore alternative assets like sneakers, watches, and LEGOs.</div>', unsafe_allow_html=True)

def page_chat_ai():
    st.header("💎 Chat with AarthAI")
    st.caption("Ask about SIPs, saving tips, or analyze a stock by typing its symbol (e.g., 'analyze TCS.NS') 📈")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            ("AarthAI", {"type": "text", "content": "Hi! I'm AarthAI. Ask me anything about Indian finance or request a stock analysis."})
        ]
    
    q = st.text_input("💬 Ask AarthAI:", key="chat_input")
    
    if st.button("Send") and q:
        st.session_state.chat_history.append(("You", q))
        with st.spinner("AarthAI thinking..."):
            r = get_ai_reply(q)
        st.session_state.chat_history.append(("AarthAI", r))
        st.rerun() 
    
    start_index = 1 if len(st.session_state.chat_history) > 1 else 0
    
    for i, (s, m) in enumerate(st.session_state.chat_history[start_index:], start=start_index):
        if s == "You": 
            st.markdown(f"🧑‍💻 **You:** {m}")
        else:
            if isinstance(m, dict) and m.get("type") == "stock":
                st.markdown(f"<div class='stock-card'><div class='stock-title'>💹 Stock Insight</div><div class='stock-detail'>{m['content']}</div></div>", unsafe_allow_html=True)
                st.plotly_chart(m["chart"], width='stretch', key=f"chart_{i}")
            else:
                st.markdown(f"💎 **AarthAI:** {m['content'] if isinstance(m, dict) else m}")

def page_stock_insights():
    st.header("📊 Stock Insights — Powered by Yahoo Finance")
    symbol = st.text_input("Enter NSE Symbol (e.g. RELIANCE.NS):", "RELIANCE.NS")

    if st.button("Fetch Insights"):
        with st.spinner("Fetching data..."):
            try:
                data = fetch_stock_details(symbol)
                if not data or data["history"].empty:
                    st.error("⚠️ Could not fetch data for this stock. Try another symbol.")
                    return

                commentary, tone = analyze_trend(data["history"])
                color_emoji = "🟢" if "Uptrend" in tone else "🔴" if "Downtrend" in tone else "🟡"

                st.markdown(
                    f"""
                    <div class="stock-card">
                        <div class="stock-title">💹 {data.get('name', symbol)} ({data['symbol']})</div>
                        <div class="stock-detail">
                        Sector: <b>{data.get('sector', 'N/A')}</b><br>
                        Current Price: ₹{data.get('price', 0):,.2f}<br>
                        52W High / Low: ₹{data.get('high', 0):,.2f} / ₹{data.get('low', 0):,.2f}<br><br>
                        <b>Trend:</b> {color_emoji} {tone}<br>
                        🧠 <i>{commentary}</i>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                fig = plot_stock_chart(data["history"], symbol)
                st.plotly_chart(fig, width='stretch')

            except Exception as e:
                st.error(f"🚫 Failed to load stock data: {e}")

def page_predict_stock():
    st.header("📈 Predict Stock Prices")
    st.caption("A simple forecast for the next N days based on the last year's trend.")

    symbol = st.text_input("Enter Stock Symbol (e.g., TCS.NS):", "TCS.NS")
    days = st.slider("Predict next N days:", 5, 30, 7)

    if st.button("Predict"):
        with st.spinner("Training model and forecasting..."):
            
            df = yf.download(symbol, period="1y", progress=False, auto_adjust=True)
            if df.empty:
                st.error("⚠️ Could not fetch stock data. Try another symbol like RELIANCE.NS or INFY.NS.")
                return

            df["Day"] = np.arange(len(df))
            X = df[["Day"]]
            y = df["Close"]
            
            try:
                degree = 3
                model = make_pipeline(StandardScaler(), PolynomialFeatures(degree), LinearRegression())
                model.fit(X, y) 
            except Exception as e:
                model = LinearRegression()
                model.fit(X, y)
            
            future_days = np.arange(len(df), len(df) + days).reshape(-1, 1)
            future_pred = model.predict(future_days) 
            forecast_dates = pd.date_range(df.index[-1], periods=days + 1, freq="B")[1:]

            forecast_df = pd.DataFrame({
                "Date": forecast_dates,
                "Predicted Close (₹)": future_pred.flatten()
            })
            st.markdown("### 🔮 Forecasted Prices")
            st.dataframe(forecast_df.style.format({"Predicted Close (₹)": "₹{:.2f}"}), width='stretch')

            st.success("✅ Forecast completed successfully!")

def page_portfolio():
    st.header("💼 Smart Portfolio Tracker")
    st.caption("Add stocks you own to track your total investment and P/L. Data is not saved on refresh (yet!).")
    
    if "portfolio" not in st.session_state:
        st.session_state.portfolio = []

    with st.form("add_stock"):
        c1, c2, c3 = st.columns(3)
        ticker = c1.text_input("Stock Symbol", placeholder="e.g., RELIANCE.NS")
        qty = c2.number_input("Quantity", min_value=1, step=1)
        price = c3.number_input("Buy Price (₹)", min_value=1.0, step=0.1)
        submitted = st.form_submit_button("Add to Portfolio")

    if submitted and ticker:
        st.session_state.portfolio.append({"ticker": ticker.strip().upper(), "qty": qty, "price": price})
        st.success(f"✅ Added {ticker} to portfolio")

    if st.session_state.portfolio:
        df = pd.DataFrame(st.session_state.portfolio)
        current_vals = []
        with st.spinner("Fetching current prices..."):
            for t in df["ticker"].unique(): 
                try:
                    stock_data = yf.download(t, period="5d", progress=False, auto_adjust=True)
                    if stock_data.empty:
                        current_vals.append((t, np.nan))
                    else:
                        current_vals.append((t, stock_data["Close"].iloc[-1]))
                except Exception:
                    current_vals.append((t, np.nan))
        
        price_map = dict(current_vals)
        df["current"] = df["ticker"].map(price_map)

        df["invested"] = df["qty"] * df["price"]
        df["value"] = df["qty"] * df["current"]
        df["pnl"] = df["value"] - df["invested"]

        df_clean = df.dropna(subset=["current"])
        if df_clean.empty:
            st.warning("⚠️ No valid stock data could be fetched.")
            return

        st.dataframe(df_clean.fillna("N/A"), width='stretch')

        total_invested = df_clean["invested"].sum()
        total_value = df_clean["value"].sum()
        total_pnl = df_clean["pnl"].sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Invested", rupee_fmt(total_invested))
        c2.metric("Current Value", rupee_fmt(total_value))
        c3.metric("Net P/L", rupee_fmt(total_pnl))

        alloc = df_clean.groupby("ticker")["invested"].sum().reset_index()
        fig = px.pie(alloc, values="invested", names="ticker", title="Portfolio Allocation (₹)")
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("🪙 Add your first stock to begin tracking your portfolio.")

def page_calculators():
    st.header("🧮 Calculators — SIP & Compound")
    c = st.radio("Select", ["SIP", "Compound Interest"], horizontal=True)
    
    if c=="SIP":
        m = st.number_input("Monthly Investment ₹", min_value=1, value=1000, step=100)
        r = st.number_input("Annual Return %", min_value=0.1, max_value=50.0, value=12.0, step=0.1)
        y = st.number_input("Years", min_value=1, max_value=50, value=10, step=1)
        
        if st.button("Calculate SIP"):
            total_invested = m * 12 * y
            monthly_rate = r / 100 / 12
            n_periods = 12 * y
            fv = m * (((1 + monthly_rate)**n_periods - 1) / monthly_rate)
            st.success(f"Future Value: {rupee_fmt(fv)} | Total Invested: {rupee_fmt(total_invested)}")
            
    else: 
        p = st.number_input("Principal ₹", min_value=1, value=10000, step=1000)
        r = st.number_input("Rate %", min_value=0.1, max_value=50.0, value=8.0, step=0.1)
        y = st.number_input("Years", min_value=1, max_value=60, value=10, step=1)
        
        if st.button("Calculate Interest"):
            fv = p * ((1 + r / 100)**y) 
            interest_earned = fv - p
            st.success(f"Future Value: {rupee_fmt(fv)} | Total Interest: {rupee_fmt(interest_earned)}")

def page_planner():
    st.header("🧭 Financial Planner")
    st.caption("Allocate your monthly income based on your financial goals.")
    
    i = st.number_input("Monthly Income (Take-Home) ₹", min_value=1000, value=50000, step=1000)
    
    r = st.radio("What's your financial priority?",
                 ["Low Risk (Balanced)", 
                  "Medium Risk (Growth-Focused)", 
                  "High Risk (Aggressive Saver)"], 
                 index=0)
    
    if i > 0:
        alloc_map = {
            "Low Risk (Balanced)":           [0.50, 0.20, 0.15, 0.10, 0.05], 
            "Medium Risk (Growth-Focused)": [0.45, 0.20, 0.20, 0.10, 0.05],
            "High Risk (Aggressive Saver)":  [0.40, 0.15, 0.30, 0.10, 0.05],
        }
        allocs = alloc_map[r]
        labels = ["Needs (Rent, Bills, EMI)", 
                  "Wants (Food, Fun, Shopping)", 
                  "Investments (SIPs, Stocks)",
                  "Emergency Fund (Savings)",
                  "Insurance (Health, Term)"]
        
        vals = [x * i for x in allocs]
        
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c4, c5, _ = st.columns(3)
        
        c1.metric(labels[0], rupee_fmt(vals[0]))
        c2.metric(labels[1], rupee_fmt(vals[1]))
        c3.metric(labels[2], rupee_fmt(vals[2]))
        c4.metric(labels[3], rupee_fmt(vals[3]))
        c5.metric(labels[4], rupee_fmt(vals[4]))
        
        st.markdown("---")
        
        fig = px.pie(values=vals, names=labels, title="Your Monthly Income Breakdown", hole=0.3)
        fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
        fig.update_layout(template="plotly_dark", legend=dict(orientation="h", yanchor="bottom", y=-0.3))
        st.plotly_chart(fig, width='stretch')

def page_side_investments():
    st.header("🎯 Side Investments — Modern Assets")
    st.caption("Explore alternative assets that can diversify your portfolio beyond stocks.")

    st.markdown("<div class='learn-header'>💎 Physical & Tangible</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("👟 Sneakers")
        st.markdown("Limited edition sneakers from brands like Nike (Jordans) and Adidas (Yeezy) can have a massive resale market. Scarcity and cultural hype drive their value.")
        
    with c2:
        st.subheader("⌚ Luxury Watches")
        st.markdown("Certain models from brands like Rolex, Patek Philippe, and Audemars Piguet are seen as 'status assets' that often appreciate in value over time due to brand power and limited supply.")

    with c3:
        st.subheader("🧱 LEGO Sets")
        st.markdown("A surprisingly profitable niche. Retired, sealed LEGO sets (especially from popular franchises like Star Wars or modular buildings) can appreciate significantly for collectors.")

    c4, c5, c6 = st.columns(3)

    with c4:
        st.subheader("🍷 Fine Wine & Whiskey")
        st.markdown("Collectibles you can drink! Bottles from renowned vineyards (like Bordeaux) or rare single malt whiskeys can gain value as they age and become scarcer.")

    with c5:
        st.subheader("🎨 Art & Collectibles")
        st.markdown("From trading cards (like Pokémon) to contemporary art (like Banksy or KAWS prints), physical collectibles are a passion-driven market. High risk, high reward.")
        
    with c6:
        st.subheader("🌟 Gold (SGBs)")
        st.markdown("A traditional store of value. In India, **Sovereign Gold Bonds (SGBs)** are a smart way to invest digitally, earning interest on top of gold prices, all tax-free on maturity.")

    st.markdown("<div class='learn-header'>🏠 Digital & Real Assets</div>", unsafe_allow_html=True)
    c7, c8, c9 = st.columns(3)

    with c7:
        st.subheader("🏠 Real Estate (REITs)")
        st.markdown("The classic wealth-builder. For those who can't buy a whole property, **REITs (Real Estate Investment Trusts)** let you buy 'shares' of a large portfolio of properties (like office parks) on the stock market.")

    with c8:
        st.subheader("💻 Digital Assets")
        st.markdown("This includes buying/selling niche websites, valuable domain names, or even investing in your own high-value skills (like AI, coding) which are the ultimate compounding asset.")

    with c9:
        st.subheader("🤝 P2P Lending")
        st.markdown("Peer-to-Peer lending platforms (like Cred Mint, LenDenClub) let you lend your money directly to other individuals for interest rates often higher than FDs. Comes with higher risk of default.")

def page_learn():
    st.header("📚 FINSIGHTS Academy — Learn & Quiz")
    st.subheader("Your one-stop shop to level up your financial literacy.")

    st.markdown("<div class='learn-header'>🎓 Beginner's Bootcamp</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.expander("📘 What is a Stock?"):
            st.markdown("A stock (or share) represents **ownership** in a company. When you buy a share of Reliance, you own a tiny piece of Reliance. You make money if the company's value grows (and the stock price goes up) or if it pays dividends (a share of the profits).")
    with c2:
        with st.expander("📗 What is Inflation? (The Enemy)"):
            st.markdown("Inflation is the rate at which prices for goods and services rise. If inflation is 7%, your money in a savings account earning 3% is **losing 4%** of its purchasing power every year. **Investing is the only way to beat inflation.**")
    with c3:
        with st.expander("📙 What is Compounding? (The Friend)"):
            st.markdown("Compounding is when your investment returns start earning their own returns. It's like a 'snowball effect.' A ₹10,000 investment growing at 12% for 30 years becomes **₹3,00,000**. The last few years of growth are bigger than the first 20.")
            
    c4, c5, c6 = st.columns(3)
    with c4:
        with st.expander("📘 What is a Mutual Fund?"):
            st.markdown("A mutual fund is a **basket of stocks** managed by a professional. When you invest in a fund, you're instantly diversified (you own small pieces of many companies). A **SIP** (Systematic Investment Plan) is just a way to invest a fixed amount into a mutual fund every month.")
    with c5:
        with st.expander("📗 What is an Index Fund?"):
            st.markdown("An index fund is a special, low-cost mutual fund. Instead of a manager *picking* stocks, it simply **buys all the stocks in an index**, like the Nifty 50 (India's top 50 companies). This is a simple, proven, and highly-recommended way to invest for the long term.")
    with c6:
        with st.expander("📙 The Golden Rule: Risk vs. Reward"):
            st.markdown("""
            This is the most important concept in finance:
            * **Higher Potential Reward = Higher Risk.** There is no "high return, safe" investment.
            * **Risk Management** isn't about *avoiding* risk, it's about *understanding* it.
            * **Diversification** (not putting all your eggs in one basket) is the #1 way to manage risk.
            """)

    st.markdown("<div class='learn-header'>🛡️ Protecting Your Wealth</div>", unsafe_allow_html=True)
    c_ins1, c_ins2 = st.columns(2)
    with c_ins1:
        with st.expander("🩺 What is Health Insurance?"):
            st.markdown("This is **non-negotiable**. Health insurance covers your medical bills (hospitalization, treatments, etc.). A single medical emergency can wipe out *years* of savings. This is the **first** financial product you should buy before you even start investing.")
    with c_ins2:
        with st.expander("❤️ What is Term Life Insurance?"):
            st.markdown("Term insurance is **pure life cover**. You pay a small premium (e.g., ₹1000/month) and if you pass away, your family gets a large payout (e.g., ₹1 Crore). It's a safety net to ensure your family is financially stable without you. **It's a must-have if you have dependents.**")

    st.markdown("<div class='learn-header'>🚀 Next Level Concepts</div>", unsafe_allow_html=True)
    c7, c8, c9 = st.columns(3)
    with c7:
        with st.expander("🚀 What is an IPO?"):
            st.markdown("An Initial Public Offering (IPO) is when a private company 'goes public' by selling its stocks to everyone for the first time. It's the company's big debut on the stock market, like Zomato or Nykaa's IPOs.")
    with c8:
        with st.expander("🚀 Reading a Stock (The Basics)"):
            st.markdown("""
            * **Market Cap:** The total value of a company (Share Price x Total Shares). A 'Large-Cap' company is a huge, stable business (e.g., HDFC Bank).
            * **P/E Ratio (Price-to-Earnings):** A quick check if a stock is 'expensive' or 'cheap' compared to its profits. A high P/E (like 80) means investors expect high future growth.
            """)
    with c9:
        with st.expander("🚀 Equity vs. Debt (Stocks vs. Bonds)"):
            st.markdown("""
            * **Equity (Stocks):** You are an **owner**. You get profits (high risk, high reward).
            * **Debt (BODs/FDs):** You are a **lender**. You get guaranteed interest (low risk, low reward).
            A good portfolio needs a mix of both.
            """)
    
    st.markdown("---")
    
    st.subheader("🧠 Knowledge Check!")
    st.caption("Test your new skills. Your answers aren't saved.")

    q1 = st.radio(
        "**Q1: What is the *first* financial product you should get before investing?**",
        ["A high-risk stock", "Health Insurance", "A P/E Ratio"],
        index=None, key="q1"
    )
    
    q2 = st.radio(
        "**Q2: What is a Nifty 50 Index Fund?**",
        ["A fund that invests in the 50 biggest companies in the world", "A low-cost fund that buys all 50 stocks in the Nifty 50 index", "A high-risk fund managed by a star investor"],
        index=None, key="q2"
    )

    q3 = st.radio(
        "**Q3: The 'snowball effect' of investing, where your returns earn more returns, is called:**",
        ["An IPO", "Compounding", "Market Cap"],
        index=None, key="q3"
    )

    if st.button("Submit Quiz"):
        correct = 0
        total = 3
        
        if q1 == "Health Insurance":
            correct += 1; st.success("Q1: Correct! Protecting your savings with health insurance is the #1 priority.")
        else:
            st.error("Q1: Incorrect. Health insurance is the first step to protect your finances from a medical emergency.")
            
        if q2 == "A low-cost fund that buys all 50 stocks in the Nifty 50 index":
            correct += 1; st.success("Q2: Correct! It's a simple, diversified, and low-cost way to own the market.")
        else:
            st.error("Q2: Incorrect. An index fund automatically buys all the stocks in a specific index.")
            
        if q3 == "Compounding":
            correct += 1; st.success("Q3: Correct! Compounding is the 8th wonder of the world.")
        else:
            st.error("Q3: Incorrect. The 'snowball effect' is called compounding.")

        st.markdown("---")
        if correct == total:
            st.balloons()
            st.markdown(f"## Your Score: {correct} / {total} — Perfect! You're a finance whiz! 🏆")
        elif correct > 0:
            st.markdown(f"## Your Score: {correct} / {total} — Great job! 👏")
        else:
            st.markdown(f"## Your Score: {correct} / {total} — Keep learning! 📚")

pages = {
    "🏠 Home": page_home,
    "💬 Chat with AarthAI": page_chat_ai,
    "📊 Stock Insights": page_stock_insights,
    "📈 Predict Stock Trends": page_predict_stock,
    "💼 Portfolio": page_portfolio,
    "🧮 Calculators": page_calculators,
    "🧭 Planner": page_planner,
    "🎯 Side Investments": page_side_investments,
    "📚 Learn": page_learn
}

def main():
    pages.get(st.session_state.page, page_home)()

if __name__ == "__main__":

    main()
