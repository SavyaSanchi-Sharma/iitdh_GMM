import numpy as np 
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
class GMMBayesClassifier:
        def __init__(self, n_components_list, random_state=42):
            self.n_components_list = n_components_list
            self.random_state = random_state

        def initialize_gmm_with_kmeans(self, X, n_components):
            kmeans = KMeans(n_clusters=n_components, random_state=self.random_state)
            labels = kmeans.fit_predict(X)
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=self.random_state)
            gmm.means_ = kmeans.cluster_centers_
            gmm.weights_ = np.bincount(labels) / len(X)
            log_likelihoods = []
            for _ in range(100):  
                gmm.fit(X)
                log_likelihoods.append(gmm.lower_bound_)
            gmm.fit(X)  
            return gmm,log_likelihoods
        def fit_and_predict(self, X_train, y_train, X_test):
            results = {}
            gmms = {}
            log_likelihoods = {}
            classes = np.unique(y_train)

            for n_components in self.n_components_list:
                print(f"Evaluating GMM with {n_components} mixtures...")
                gmms_per_class = {}
                priors = {}
                log_likelihoods_per_class = []

                for c in classes:
                    X_c = X_train[y_train == c]
                    gmm, log_likelihood = self.initialize_gmm_with_kmeans(X_c, n_components)
                    gmms_per_class[c] = gmm
                    priors[c] = len(X_c) / len(X_train)
                    log_likelihoods_per_class.append(log_likelihood)

                gmms[n_components] = gmms_per_class
                log_likelihoods[n_components] = np.mean(log_likelihoods_per_class, axis=0)

                log_probs = []
                for c in classes:
                    log_prob = gmms_per_class[c].score_samples(X_test) + np.log(priors[c])
                    log_probs.append(log_prob)

                log_probs = np.array(log_probs).T
                y_pred = classes[np.argmax(log_probs, axis=1)]
                results[n_components] = y_pred

            return results, gmms, log_likelihoods

        def evaluate_metrics(self, y_true, y_pred):
            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average=None)
            recall = recall_score(y_true, y_pred, average=None)
            f1 = f1_score(y_true, y_pred, average=None)

            mean_precision = precision_score(y_true, y_pred, average='macro')
            mean_recall = recall_score(y_true, y_pred, average='macro')
            mean_f1 = f1_score(y_true, y_pred, average='macro')

            metrics = {
                'accuracy': acc,
                'precision': precision,
                'mean_precision': mean_precision,
                'recall': recall,
                'mean_recall': mean_recall,
                'f1': f1,
                'mean_f1': mean_f1
            }

            return metrics

        def plot_confusion_matrix(self, y_true, y_pred):
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.show()
        def plot_log_likelihood(self, log_likelihoods):
            for n_components, log_likelihood in log_likelihoods.items():
                plt.plot(log_likelihood, label=f'{n_components} mixtures')
            plt.title('Iterations vs Log Likelihood')
            plt.xlabel('Iterations')
            plt.ylabel('Log Likelihood')
            plt.legend()
            plt.show() 