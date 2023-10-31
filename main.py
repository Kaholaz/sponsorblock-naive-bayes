from bayes import NaiveBayesClassifier

training_data = [
    ("spam email content 1", "spam"),
    ("ham email content 1", "ham"),
    ("spam email content 2", "spam"),
    ("ham email content 2", "ham"),
]

testing_data = [
    "spam email content 3",
    "ham spam spam",
]


def main():
    bayes = NaiveBayesClassifier(training_data=training_data)
    bayes.train()
    bayes.test(testing_data=testing_data)


if __name__ == "__main__":
    main()
