💹 FINSIGHTS AI

FINSIGHTS AI is an all-in-one financial companion designed for Gen-Z investors. Built entirely in Python with Streamlit, this app integrates an AI chatbot, machine learning-based stock prediction, a portfolio tracker, and a comprehensive learning academy into a single, easy-to-use interface.

This app was built for a hackathon, showcasing the power of integrating smart AI tools with real-world financial data.

🚀 Live Demo (Placeholder)

(Insert a GIF or screenshot of the app here.)

✨ Key Features

💬 Smart AI Chatbot (AarthAI): A conversational AI assistant (powered by GPT-4o-mini) that can answer finance questions. It's also a "tool-user": ask it to "analyze RELIANCE.NS" and it will bypass the LLM, fetch real-time data using yfinance, and present a full stock analysis card directly in the chat.

📊 Real-Time Stock Insights: Get instant data on any NSE/BSE stock, including 6-month charts, 52-week high/low, and an AI-generated momentum analysis (Uptrend, Downtrend, or Neutral).

📈 ML Stock Prediction: A predictive model (Polynomial Regression) that forecasts the next 1-30 days of a stock's price, trained on the last year of data.

💼 Portfolio Tracker: A session-based portfolio where you can add your stock holdings (symbol, quantity, buy price) to see your total investment, current value, and net profit/loss in real-time.

🧭 Financial Planner: An interactive tool to allocate your monthly income. It uses a risk-based model (Low, Medium, High) to divide your earnings between Needs, Wants, Investments, an Emergency Fund, and Insurance.

📚 Learn Academy & Quiz: A rich, content-filled learning hub with bite-sized lessons on everything from "What is a Stock?" and "Health Insurance" to "Real Estate (REITs)" and "P/E Ratios". It includes an interactive quiz to test your knowledge.

🎯 Side Investment Explorer: A text-only guide to modern, alternative assets like sneakers, luxury watches, LEGOs, fine wine, and digital assets (REITs, P2P Lending).

🧮 Financial Calculators: Quick and easy tools for calculating SIP (Systematic Investment Plan) future value and Compound Interest.

🛠️ Tech Stack

Core: Python

Web Framework: Streamlit

Data & ML: Pandas, NumPy, Scikit-learn (Polynomial Regression, StandardScaler)

Financial Data: yfinance (Yahoo Finance)

AI Chatbot: OpenAI (GPT-4o-mini)

Plotting: Plotly Express

Environment: python-dotenv

Setup & Installation

To run this project locally, follow these steps:

1. Clone the Repository

git clone [https://github.com/YOUR_USERNAME/finsights-ai.git](https://github.com/YOUR_USERNAME/finsights-ai.git)
cd finsights-ai


2. Create a Virtual Environment

# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate


3. Install Dependencies
All required libraries are listed in requirements.txt.

pip install -r requirements.txt


4. Set Up Your API Key
The chatbot requires an OpenAI API key.

Create a file named .env in the root directory.

Add your API key to this file:

OPENAI_API_KEY="sk-YourSecretKeyGoesHere"


5. Run the App
You're all set! Run the following command in your terminal:

streamlit run finsights_app_clean.py


The app will open in your browser at http://localhost:8501.

📂 File Structure

The entire application is contained in a single file for simplicity.

finsights-ai/
├── finsights_app_clean.py   # The main Streamlit app
├── requirements.txt         # All Python dependencies
├── .env                     # Your API key (local only, do not commit)
└── README.md


📄 License

This project is distributed under the MIT License. See LICENSE for more information.

Acknowledgements

Built by Abhiram M.

Data via Yahoo Finance

App framework by Streamlit
