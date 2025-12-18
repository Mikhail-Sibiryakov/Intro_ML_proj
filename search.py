import pickle
import numpy as np
from scipy.spatial.distance import cosine
import fasttext

with open('image_index.pkl', 'rb') as f:
    image_index = pickle.load(f)


def search(query, ft_model, top_k=5):
    query_words = query.lower().split()
    query_vecs = [ft_model.get_word_vector(w) for w in query_words]
    query_vec = np.mean(query_vecs, axis=0)

    results = []

    for item in image_index:
        vec, img_path, prob = item

        semantic_sim = 1 - cosine(query_vec, vec)

        final_score = semantic_sim * prob

        results.append((img_path, final_score))

    results.sort(key=lambda x: x[1], reverse=True)

    unique_results = []
    seen_paths = set()
    for path, score in results:
        if path not in seen_paths:
            unique_results.append((path, score))
            seen_paths.add(path)
        if len(unique_results) >= top_k:
            break

    return unique_results


ft_model = fasttext.load_model('./model.bin')

user_query = ""
print("I am ready...")
if __name__ == "__main__":
    while True:
        user_query = input()
        if user_query == 'exit':
            break
        top_images = search(user_query, ft_model)
        print(f"Результаты поиска по запросу '{user_query}':")
        for path, score in top_images:
            print(f"Схожесть: {score:.4f} -> {path}")
