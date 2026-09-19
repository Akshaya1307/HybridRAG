from src.generator import generate_answer

question = "What is machine learning?"

chunks = [
    {
        "page_number": 1,
        "text": "Machine learning is a method where computers learn patterns from data."
    }
]

answer = generate_answer(
    question,
    chunks
)

print("\nANSWER:")
print(answer)