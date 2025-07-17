import pandas as pd
import openai
from io import StringIO
from fuzzywuzzy import fuzz
import os

csv_data = """prompt,response
"what is psoriasis and what are its common symptoms?","psoriasis is a chronic autoimmune condition that results in the overproduction of skin cells. this overproduction leads to patches of thick, red skin covered with silvery scales. common symptoms include red patches of skin covered with thick, silvery scales, small scaling spots (commonly seen in children), dry and cracked skin that may bleed, itching, burning, or soreness, thickened, pitted, or ridged nails, and swollen and stiff joints."
"what is the etiology of acne?","acne is primarily caused by the overproduction of oil; blocked hair follicles that don't allow oil to leave the pore, which may cause a clogged pore; bacteria called propionibacterium acnes that can grow inside the hair follicles and cause inflammation; and hormonal changes, particularly during puberty and menstruation. other factors like certain medications, diet, and stress can also contribute to the development of acne."
"what are the recommended medications for atopic dermatitis?","there are several medications available for the treatment of atopic dermatitis. topical corticosteroids are often the first line of treatment. these can help to reduce inflammation and itching. other options include topical calcineurin inhibitors, which affect the immune system and help to maintain normal skin texture and reduce flare-ups. in severe cases, systemic drugs that work throughout the body may be usedनि, or oral drugs to control inflammation. light therapy, using a machine that emits uvb light, is another treatment option for severe eczema."
# Truncated for brevity; full CSV data is assumed to be available as provided
"""


df = pd.read_csv(StringIO(csv_data))
qa_dict = dict(zip(df['prompt'], df['response']))

# Set up OpenAI API key (replace with your actual key or use environment variable)
# put your api key here   

def get_csv_answer_exact(question):
    """Check for exact match in CSV data."""
    return qa_dict.get(question.strip(), None)

def get_csv_answer_fuzzy(question, threshold=80):
    """Check for fuzzy match in CSV data."""
    for q, a in qa_dict.items():
        if fuzz.ratio(question.lower(), q.lower()) > threshold:
            return a
    return None

def get_chatgpt_response(question):
    """Get response from OpenAI API."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in dermatology and skin health."},
                {"role": "user", "content": question}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with OpenAI API: {e}"

def chatbot_response(user_input):
    """Generate response by checking CSV first, then falling back to OpenAI API."""
    # Try exact match
    csv_answer = get_csv_answer_exact(user_input)
    if csv_answer:
        return csv_answer
    
    # Try fuzzy match
    csv_answer = get_csv_answer_fuzzy(user_input)
    if csv_answer:
        return csv_answer
    
    # Fallback to OpenAI API
    return get_chatgpt_response(user_input)

def main():
    """Run the chatbot in a terminal interface."""
    print("Welcome to the Skin Health Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = chatbot_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()