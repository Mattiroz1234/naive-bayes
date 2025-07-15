# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import make_pipeline
#
# # 1. date
# texts = [
#     "Free money now!!!",         # spam
#     "Call this number for a prize",  # spam
#     "Hi, how are you?",          # ham
#     "Let's meet tomorrow",       # ham
#     "Win cash by entering now",  # spam
#     "Are you coming to the party?" # ham
# ]
#
# labels = ['spam', 'spam', 'ham', 'ham', 'spam', 'ham']
#
# # 2. pipline vectorization & Naive Bayes
# model = make_pipeline(CountVectorizer(), MultinomialNB())
#
# # 3. training model
# model.fit(texts, labels)
#
# # 4. classification a new message:
# test_message = "Free prize just for you"
# prediction = model.predict([test_message])[0]
#
# print(f"message: '{test_message}' classified as: {prediction}")
#
# probs = model.predict_proba([test_message])[0]
# print(f"probability:")
# print(f"spam: {probs[model.classes_.tolist().index('spam')]}")
# print(f"ham: {probs[model.classes_.tolist().index('ham')]}")
