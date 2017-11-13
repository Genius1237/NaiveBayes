#include <iostream>
#include <string>
#include <ctime>
#include <iomanip>

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
	cout << "``````````````````````````\n";
	NaiveBayesClassifier nb1(neg_max, pos_min, train_bow_file, vocab_file, "");

	cout << "Without binarization\n";
	clock_t t = clock();
	nb1.test(test_bow_file, false);
	cout << "Time taken: " << setprecision(4) << ((clock() - t) / static_cast<double>(CLOCKS_PER_SEC)) << " seconds\n";

	cout << "\nWith binarization\n";
	t = clock();
	nb1.test(test_bow_file, true);
	cout << "Time taken: " << setprecision(4) << ((clock() - t) / static_cast<double>(CLOCKS_PER_SEC)) << " seconds\n";

	// after removing stopwords
	cout << "\nAfter removing stopwords\n";
	cout << "````````````````````````\n";
	NaiveBayesClassifier nb2(neg_max, pos_min, train_bow_file, vocab_file, sw_file);

	cout << "Without binarization\n";
	t = clock();
	nb2.test(test_bow_file, false);
	cout << "Time taken: " << setprecision(4) << ((clock() - t) / static_cast<double>(CLOCKS_PER_SEC)) << " seconds\n";

	cout << "\nWith binarization\n";
	t = clock();
	nb2.test(test_bow_file, true);
	cout << "Time taken: " << setprecision(4) << ((clock() - t) / static_cast<double>(CLOCKS_PER_SEC)) << " seconds\n\n";
	return 0;
}
