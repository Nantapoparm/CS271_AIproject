from flask import Flask, render_template, request, jsonify
from chat1 import fetch_website_content, extract_pdf_text, initialize_vector_store
from chat2 import llm, setup_retrieval_qa

app = Flask(__name__)

# Initialize data once
urls = ["https://oae.go.th/home"]
pdf_files = ["Data/Farmerbook1.pdf", "Data/Farming Schemes.pdf"]

print("--- Loading data, please wait... ---")
website_contents = [fetch_website_content(url) for url in urls]
pdf_texts = [extract_pdf_text(pdf_file) for pdf_file in pdf_files]

all_contents = website_contents + pdf_texts
db = initialize_vector_store(all_contents)
chain = setup_retrieval_qa(db)
print("--- AgriGenius is ready! ---")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    # ดึงข้อมูลจาก messageText (รองรับทั้ง JSON และ Form)
    if request.is_json:
        query = request.json.get('messageText', '').strip()
    else:
        query = request.form.get('messageText', '').strip()

    if not query:
        return jsonify({"answer": "Please enter a question."})

    # เช็คคำถามพิเศษ (Manual Check)
    lower_query = query.lower()
    if lower_query in ["who developed you?", "who created you?", "who made you?"]:
        return jsonify({"answer": "I was developed by Jayesh Bhandarkar."})

    try:
        # เรียกใช้งาน Chain (LangChain 0.3.x ใช้ invoke)
        response = chain.invoke({"input": query})
        return jsonify({"answer": response['answer']})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=False)
