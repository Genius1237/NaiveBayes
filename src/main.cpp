#include <iostream>
#include <string>

using namespace std;

#include "naive_bayes.h"

int main() {
	// set variables which would be used as arguments later
	int neg_max = 4, pos_min = 7;
	string train_bow_file = "../data/train.feat";
	string test_bow_file = "../data/test.feat";
	string vocab_file = "../data/imdb.vocab";
	string sw_file = "../data/sw";

	// without removing stopwords
	cout << "\nWithout removing stopwords\n";
	cout << "``````````````````````\n";
	NaiveBayesClassifier nb1(neg_max, pos_min, train_bow_file, vocab_file, "");
	cout << "Without binarization\n";
	nb1.test(test_bow_file, false);
	cout << "\nWith binarization\n";
	nb1.test(test_bow_file, true);

	// after removing stopwords
	cout << "\nAfter removing stopwords\n";
	cout << "````````````````````````\n";
	NaiveBayesClassifier nb2(neg_max, pos_min, train_bow_file, vocab_file, sw_file);
	cout << "Without binarization\n";
	nb2.test(test_bow_file, false);
	cout << "\nWith binarization\n";
	nb2.test(test_bow_file, true);
	cout << "\n";
	return 0;
}