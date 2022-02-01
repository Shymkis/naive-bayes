# Implement the Naive Bayes algorithm (varying the parameter k in Laplace smoothing;
# NOTE: the use of k here reflects the assumption that every outcome is seen an extra k times,
# as described in Slide 33 of the "SupervisedLearningNaiveBayes-SS.pptx" file,
# and does not correspond to the number of classes, as in page 12 of the cs229-notes2.pdf file)
# and evaluate it on at least one image (using the multinomial event model) and one text data set
# (using both the multivariate Bernoulli vs multinomial event models).
# Submit with your implementation files a report that describes the implementation choices
# as well as analysis and highlights of experimental results on datasets of your choice.
# You must include data you have collected for your experiments or a link to the source data
# if you used data available on the web.  Your report should be ~5 pages of 12 point,
# single column and spacing with 1" margins and should be modeled after published articles
# using appropriate use of abstract, section headers (introduction, models, experimental results,
# discussions, related work, conclusions) and citations to relevant existing research.
# Use 5-fold cross validation to determine model accuracy. Grade will depend on the following items
# - correct implementation of all functionalities (you can use existing helper routines to gather data
#   but all probability calculations should be made with code that you develop on your own): 40%
# - thorough experimental results: 25%
# - analysis of the results to highlight relative strengths and weaknesses of the 
#   multivariate Bernoulli vs multinomial event models, the use of Laplace smoothing, etc. 25%
# - professional writing style, use of appropriate references, etc. 10%
# Joe Shymanski
# Project 1: Naive Bayes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

def n_folds(N, df):
    folds = []
    for x in range(N):
        train_data = []
        test_data = []
        for i, row in df.iterrows():
            if i % N == x:
                test_data.append(row)
            else:
                train_data.append(row)
        folds.append({"train": pd.DataFrame(train_data), "test": pd.DataFrame(test_data)})
    return folds

# Faster, more efficient, more accurate than a vector of all english words
def word_counts(texts, bernoulli):
    words = []
    for text in texts:
        # Multivariate Bernoulli: only keep each unique word within a text
        if bernoulli:
            text = set(text)
        # Multinomial: keep all words, even duplicates, within a text
        words.extend(text)
    return {word: words.count(word) for word in set(words)}

# More general for other kinds of problems
def pixel_counts(p_vecs):
    pixels = [sum(p_vec) for p_vec in zip(*p_vecs)]
    return {i: v for i, v in enumerate(pixels)}

def train_naive_bayes_model(train_df, bernoulli, k, problem):
    # Label probabilities
    label_probs = train_df.label.value_counts()/len(train_df.index)
    # Feature counts
    if problem == "text":
        f_given_label_counts = pd.DataFrame({
            label: word_counts(train_df.loc[train_df.label == label, "f_vec"], bernoulli) for label in label_probs.keys()
        }).fillna(0)
    else:
        f_given_label_counts = pd.DataFrame({
            label: pixel_counts(train_df.loc[train_df.label == label, "f_vec"]) for label in label_probs.keys()
        }).fillna(0)
    # Laplace smoothing
    f_given_label_counts += k
    # Feature probabilities
    if problem == "text":
        f_given_label_probs = f_given_label_counts.apply(lambda col : col/sum(col))
    else:
        f_given_label_probs = f_given_label_counts.apply(lambda x : x/(train_df.label.value_counts()[x.name] + k + 1))
        # f_given_label_probs = f_given_label_counts.apply(lambda x : x/(train_df.label.value_counts()[x.name] + k*len(f_given_label_counts.index)))
    return label_probs, f_given_label_probs

def naive_bayes_predict(nb_model, f_vec, problem):
    label_probs, f_given_label_probs = nb_model
    label_values = {}
    for label in label_probs.keys():
        # Slowest segment of all code
        if problem == "text":
            probs = [f_given_label_probs.loc[f, label] for f in f_vec if f in f_given_label_probs.index]
        else:
            probs = [f_given_label_probs.loc[i, label] if f else 1 - f_given_label_probs.loc[i, label] for i, f in enumerate(f_vec)]
        probs.append(label_probs[label])
        label_values[label] = sum(np.log(probs))
    return max(label_values, key=label_values.get)

def prediction_accuracy(df):
    correct = sum(df.apply(lambda row : int(row.prediction == row.label), axis=1))
    return correct / len(df.index) * 100

def main(df, fr, N, bernoulli, k, problem):
    print(problem, ", Fraction of data: ", fr, ", Bernoulli?: ", bernoulli, ", k: ", k, sep="")

    start = time.time()

    # Shuffle data
    df = df.sample(frac=fr).reset_index(drop=True)

    # N-fold cross-validation
    folds = n_folds(N, df)
    train_accuracy = test_accuracy = 0
    for fold in folds:
        train_df = fold["train"]
        test_df = fold["test"]

        # Training
        nb_model = train_naive_bayes_model(train_df, bernoulli, k, problem)

        # Make predictions on training set and testing set independently
        train_df["prediction"] = train_df.apply(lambda row : naive_bayes_predict(nb_model, row.f_vec, problem), axis=1)
        test_df["prediction"] = test_df.apply(lambda row : naive_bayes_predict(nb_model, row.f_vec, problem), axis=1)

        # Calculate accuracies for both sets of predictions
        train_accuracy += prediction_accuracy(train_df)
        test_accuracy += prediction_accuracy(test_df)

    end = time.time()
    avg_time = round((end - start) / N, 2)

    avg_train = round(train_accuracy / N, 2)
    avg_test = round(test_accuracy / N, 2)
    print("Average training set accuracy: " + str(avg_train) + "%")
    print("Average testing set accuracy: " + str(avg_test) + "%")

    print("Average time elapsed: " + str(avg_time))
    return avg_train, avg_test, avg_time

if __name__ == "__main__":
    # https://www.kaggle.com/uciml/sms-spam-collection-dataset
    df1 = pd.read_csv('spam.csv', usecols=["v1", "v2"], encoding="latin-1").rename(columns={"v1": "label", "v2": "f_vec"})
    df1["f_vec"] = df1.apply(lambda row : row.f_vec.split(), axis=1)

    # https://www.kaggle.com/c/digit-recognizer
    df2 = pd.read_csv('digits.csv')
    df2["f_vec"] = df2.drop("label", axis=1).apply(lambda x : x//128).values.tolist()
    df2 = df2[["label", "f_vec"]]

    # main(df1, fr=1, N=5, bernoulli=True, k=1, problem="text")
    # main(df1, fr=1, N=5, bernoulli=False, k=1, problem="text")
    # main(df2, fr=.01, N=5, bernoulli=False, k=1, problem="image")

    # y1 = []
    # y2 = []
    # y3 = []
    # x = []
    # for i in range(5, 26, 2):
    #     i/=10000
    #     avg_train, avg_test, avg_time = main(df2, fr=i , N=5, bernoulli=False, k=1, problem="image")
    #     y1.append(avg_train)
    #     y2.append(avg_test)
    #     y3.append(avg_time)
    #     x.append(i)
    # plot1 = plt.figure(1)
    # plt.plot(x, y1, label="Average Train Accuracy")
    # plt.plot(x, y2, label="Average Test Accuracy")
    # plt.title("Accuracy vs. Fraction of Image Data Used, k+1 in Denominator")
    # plt.xlabel("Fraction of Data")
    # plt.ylabel("Accuracy (%)")
    # plt.legend()

    # plot2 = plt.figure(2)
    # plt.plot(x, y3, label="Average Time Elapsed")
    # plt.title("Time Elapsed vs. Fraction of Image Data Used, k+1 in Denominator")
    # plt.xlabel("Fraction of Data")
    # plt.ylabel("Time Elapsed (s)")

    # plt.show()
